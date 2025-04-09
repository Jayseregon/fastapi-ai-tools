import logging
import random
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query

from src.models.neo4j_infos_models import (
    Company,
    CompanyStat,
    Employment,
    ErrorResponse,
    Neo4jStatus,
    Person,
    PersonConnection,
    PopulationResult,
    QueryParameters,
    QueryResponse,
)
from src.security.rateLimiter.depends import RateLimiter
from src.services.db import neo4j_service

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/v1/neo4j-infos", tags=["Neo4j"])


@router.get("/ping", response_model=Neo4jStatus)
async def test_neo4j(
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10))
) -> Neo4jStatus:
    """Test connectivity to the Neo4j graph database.

    This endpoint performs a simple query to verify that the Neo4j database
    is accessible and functioning correctly. Useful for system health checks
    and ensuring the graph database component is operational.

    Args:
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        A Neo4jStatus object containing the response value from Neo4j

    Raises:
        HTTPException: If Neo4j service is unavailable
    """
    try:
        # Run a simple query to test connection
        driver = neo4j_service()
        with driver.session() as session:
            result = session.run("RETURN 1 AS number")
            record = result.single()
        return Neo4jStatus(neo4j_response=record["number"] if record else None)
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {str(e)}")
        raise HTTPException(
            status_code=503, detail=f"Neo4j database is not available: {str(e)}"
        )


@router.get("/fake/populate", response_model=PopulationResult)
async def populate_neo4j(
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10))
) -> PopulationResult:
    """Populate Neo4j with fake data (people, companies, and relationships).

    This endpoint creates sample data in the Neo4j database, including person nodes,
    company nodes, and various relationships between them. Useful for testing graph
    capabilities and demonstrating knowledge graph structure for RAG systems.

    Args:
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        A PopulationResult object with details about created entities

    Raises:
        HTTPException: If data population fails
    """
    try:
        # Sample data
        people = [
            {"name": "Alice Johnson", "age": 32, "role": "Developer"},
            {"name": "Bob Smith", "age": 45, "role": "Manager"},
            {"name": "Charlie Davis", "age": 28, "role": "Designer"},
            {"name": "Diana Miller", "age": 38, "role": "CTO"},
            {"name": "Edward Wilson", "age": 25, "role": "Intern"},
        ]

        companies = [
            {"name": "Tech Solutions", "industry": "Software", "founded": 2010},
            {"name": "DataCorp", "industry": "Data Analytics", "founded": 2015},
            {"name": "WebFront", "industry": "Web Development", "founded": 2018},
        ]

        driver = neo4j_service()
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Create people nodes
            for person in people:
                session.run(
                    "CREATE (p:Person {name: $name, age: $age, role: $role})",
                    name=person["name"],
                    age=person["age"],
                    role=person["role"],
                )

            # Create company nodes
            for company in companies:
                session.run(
                    "CREATE (c:Company {name: $name, industry: $industry, founded: $founded})",
                    name=company["name"],
                    industry=company["industry"],
                    founded=company["founded"],
                )

            # Create KNOWS relationships between people
            for i in range(len(people)):
                for j in range(i + 1, len(people)):
                    if random.random() > 0.3:  # 70% chance of creating a relationship
                        session.run(
                            """
                            MATCH (p1:Person {name: $name1})
                            MATCH (p2:Person {name: $name2})
                            CREATE (p1)-[r:KNOWS {since: $since}]->(p2)
                            """,
                            name1=people[i]["name"],
                            name2=people[j]["name"],
                            since=2020 + random.randint(0, 3),
                        )

            # Create WORKS_FOR relationships
            for person in people:
                company = random.choice(companies)
                session.run(
                    """
                    MATCH (p:Person {name: $person_name})
                    MATCH (c:Company {name: $company_name})
                    CREATE (p)-[r:WORKS_FOR {position: $position, joined: $joined}]->(c)
                    """,
                    person_name=person["name"],
                    company_name=company["name"],
                    position=person["role"],
                    joined=2018 + random.randint(0, 5),
                )

        return PopulationResult(
            message="Database populated with fake data",
            people=len(people),
            companies=len(companies),
        )
    except Exception as e:
        logger.error(f"Error populating Neo4j: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to populate Neo4j: {str(e)}"
        )


@router.get("/fake/query", response_model=Union[QueryResponse, ErrorResponse])
async def query_neo4j(
    query_type: str = Query("all_people", description="Type of query to run"),
    company: Optional[str] = Query(None, description="Company name for filtering"),
    person: Optional[str] = Query(None, description="Person name for filtering"),
    limit: int = Query(10, description="Max number of results to return"),
    rate: Optional[None] = Depends(RateLimiter(times=3, seconds=10)),
) -> Union[QueryResponse, ErrorResponse]:
    """Query Neo4j database with different query types.

    This endpoint provides a flexible interface to the Neo4j graph database,
    allowing various query patterns based on the 'query_type' parameter.
    Enables exploration of graph relationships for knowledge-based AI systems.

    Available query types:
    - all_people: Lists all person nodes
    - all_companies: Lists all company nodes
    - employees_by_company: Shows employees of a specific company
    - person_network: Shows a person's connections
    - company_stats: Shows statistics about companies and employees

    Args:
        query_type: The type of query to execute
        company: Company name filter (required for employees_by_company)
        person: Person name filter (required for person_network)
        limit: Maximum number of results to return
        rate: Rate limiter dependency to prevent abuse (3 requests per 10 seconds)

    Returns:
        Either a QueryResponse with the requested data or an ErrorResponse
        if the query parameters are invalid

    Raises:
        HTTPException: If the query fails to execute
    """
    try:
        typed_results: List[
            Union[
                Person,
                Company,
                Employment,
                PersonConnection,
                CompanyStat,
                Dict[str, Any],
            ]
        ] = []

        driver = neo4j_service()
        with driver.session() as session:
            if query_type == "all_people":
                result = session.run(
                    "MATCH (p:Person) RETURN p.name AS name, p.age AS age, p.role AS role LIMIT $limit",
                    limit=limit,
                )
                typed_results = [
                    Person(name=record["name"], age=record["age"], role=record["role"])
                    for record in result
                ]

            elif query_type == "all_companies":
                result = session.run(
                    "MATCH (c:Company) RETURN c.name AS name, c.industry AS industry, c.founded AS founded LIMIT $limit",
                    limit=limit,
                )
                typed_results = [
                    Company(
                        name=record["name"],
                        industry=record["industry"],
                        founded=record["founded"],
                    )
                    for record in result
                ]

            elif query_type == "employees_by_company":
                if not company:
                    return ErrorResponse(
                        error="Company parameter is required for this query type"
                    )

                result = session.run(
                    """
                    MATCH (p:Person)-[r:WORKS_FOR]->(c:Company {name: $company})
                    RETURN p.name AS name, p.role AS role, r.joined AS joined_year
                    LIMIT $limit
                    """,
                    company=company,
                    limit=limit,
                )
                typed_results = [
                    Employment(
                        name=record["name"],
                        role=record["role"],
                        joined_year=record["joined_year"],
                    )
                    for record in result
                ]

            elif query_type == "person_network":
                if not person:
                    return ErrorResponse(
                        error="Person parameter is required for this query type"
                    )

                result = session.run(
                    """
                    MATCH (p:Person {name: $person})-[r:KNOWS]-(other:Person)
                    RETURN other.name AS name, other.role AS role, r.since AS knows_since
                    LIMIT $limit
                    """,
                    person=person,
                    limit=limit,
                )
                typed_results = [
                    PersonConnection(
                        name=record["name"],
                        role=record["role"],
                        knows_since=record["knows_since"],
                    )
                    for record in result
                ]

            elif query_type == "company_stats":
                result = session.run(
                    """
                    MATCH (c:Company)<-[r:WORKS_FOR]-(p:Person)
                    RETURN c.name AS company,
                        COUNT(p) AS employee_count,
                        AVG(p.age) AS avg_employee_age
                    LIMIT $limit
                    """,
                    limit=limit,
                )
                typed_results = [
                    CompanyStat(
                        company=record["company"],
                        employee_count=record["employee_count"],
                        avg_employee_age=record["avg_employee_age"],
                    )
                    for record in result
                ]

            else:
                return ErrorResponse(error="Unknown query type")

        return QueryResponse(
            query_type=query_type,
            parameters=QueryParameters(company=company, person=person, limit=limit),
            result_count=len(typed_results),
            results=typed_results,
        )
    except Exception as e:
        logger.error(f"Error executing Neo4j query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute query: {str(e)}"
        )
