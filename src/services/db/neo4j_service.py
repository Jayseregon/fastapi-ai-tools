import logging

from neo4j import GraphDatabase, basic_auth

from src.configs.env_config import config

logger = logging.getLogger(__name__)


class Neo4jService:
    """Service for interacting with Neo4j."""

    def __init__(self):
        """Initialize the Neo4j driver."""
        self.driver = None

    def __call__(self):
        """Get or create a Neo4j driver."""
        if self.driver:
            return self.driver

        try:
            logger.debug(f"Connecting to Neo4j at {config.NEO4J_URI}")

            auth = basic_auth(config.NEO4J_USER, config.NEO4J_PWD)
            self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=auth)

            # Test the connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.debug("Neo4j connection successful")
                else:
                    logger.warning("Neo4j connection test returned unexpected result")

            return self.driver
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {str(e)}")
            self.driver = None
            raise

    def close(self):
        """Close the Neo4j driver if it exists."""
        if self.driver:
            self.driver.close()
            self.driver = None


# Create a singleton instance
neo4j_service = Neo4jService()
