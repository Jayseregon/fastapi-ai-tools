import logging
import random
from typing import Optional

from fastapi import APIRouter, Query

from src.services.db import neo4j_service

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/v1/graph", tags=["Neo4j"])


@router.get("/neo4j")
def test_neo4j():
    # Run a simple query to test connection
    driver = neo4j_service()
    with driver.session() as session:
        result = session.run("RETURN 1 AS number")
        record = result.single()
    return {"neo4j_response": record["number"] if record else None}


@router.get("/populate")
def populate_neo4j():
    """Populate Neo4j with fake data (people, companies, and relationships)"""
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

    return {
        "message": "Database populated with fake data",
        "people": len(people),
        "companies": len(companies),
    }


@router.get("/query")
def query_neo4j(
    query_type: str = Query("all_people", description="Type of query to run"),
    company: Optional[str] = Query(None, description="Company name for filtering"),
    person: Optional[str] = Query(None, description="Person name for filtering"),
    limit: int = Query(10, description="Max number of results to return"),
):
    """Query Neo4j database with different query types"""
    results = []

    driver = neo4j_service()
    with driver.session() as session:
        if query_type == "all_people":
            result = session.run(
                "MATCH (p:Person) RETURN p.name AS name, p.age AS age, p.role AS role LIMIT $limit",
                limit=limit,
            )
            results = [
                {"name": record["name"], "age": record["age"], "role": record["role"]}
                for record in result
            ]

        elif query_type == "all_companies":
            result = session.run(
                "MATCH (c:Company) RETURN c.name AS name, c.industry AS industry, c.founded AS founded LIMIT $limit",
                limit=limit,
            )
            results = [
                {
                    "name": record["name"],
                    "industry": record["industry"],
                    "founded": record["founded"],
                }
                for record in result
            ]

        elif query_type == "employees_by_company":
            if not company:
                return {"error": "Company parameter is required for this query type"}

            result = session.run(
                """
                MATCH (p:Person)-[r:WORKS_FOR]->(c:Company {name: $company})
                RETURN p.name AS name, p.role AS role, r.joined AS joined_year
                LIMIT $limit
                """,
                company=company,
                limit=limit,
            )
            results = [
                {
                    "name": record["name"],
                    "role": record["role"],
                    "joined_year": record["joined_year"],
                }
                for record in result
            ]

        elif query_type == "person_network":
            if not person:
                return {"error": "Person parameter is required for this query type"}

            result = session.run(
                """
                MATCH (p:Person {name: $person})-[r:KNOWS]-(other:Person)
                RETURN other.name AS name, other.role AS role, r.since AS knows_since
                LIMIT $limit
                """,
                person=person,
                limit=limit,
            )
            results = [
                {
                    "name": record["name"],
                    "role": record["role"],
                    "knows_since": record["knows_since"],
                }
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
            results = [
                {
                    "company": record["company"],
                    "employee_count": record["employee_count"],
                    "avg_employee_age": record["avg_employee_age"],
                }
                for record in result
            ]

        else:
            return {"error": "Unknown query type"}

    return {
        "query_type": query_type,
        "parameters": {"company": company, "person": person, "limit": limit},
        "result_count": len(results),
        "results": results,
    }
