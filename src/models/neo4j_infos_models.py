from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Neo4jStatus(BaseModel):
    """Status response from Neo4j connection test"""

    neo4j_response: Optional[int] = Field(None, description="Response value from Neo4j")


class Person(BaseModel):
    """Person entity model"""

    name: str = Field(..., description="Person's name")
    age: int = Field(..., description="Person's age")
    role: str = Field(..., description="Person's professional role")


class Company(BaseModel):
    """Company entity model"""

    name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Company's industry sector")
    founded: int = Field(..., description="Year when company was founded")


class Employment(BaseModel):
    """Employment relationship details"""

    name: str = Field(..., description="Person's name")
    role: str = Field(..., description="Person's role at the company")
    joined_year: int = Field(..., description="Year when person joined the company")


class PersonConnection(BaseModel):
    """Connection between people"""

    name: str = Field(..., description="Person's name")
    role: str = Field(..., description="Person's role")
    knows_since: int = Field(
        ..., description="Year when the connection was established"
    )


class CompanyStat(BaseModel):
    """Company statistics"""

    company: str = Field(..., description="Company name")
    employee_count: int = Field(..., description="Number of employees")
    avg_employee_age: float = Field(..., description="Average age of employees")


class PopulationResult(BaseModel):
    """Result of populating Neo4j with sample data"""

    message: str = Field(..., description="Status message")
    people: int = Field(..., description="Number of people created")
    companies: int = Field(..., description="Number of companies created")


class QueryParameters(BaseModel):
    """Query parameters used"""

    company: Optional[str] = Field(None, description="Company name filter")
    person: Optional[str] = Field(None, description="Person name filter")
    limit: int = Field(..., description="Max number of results")


class QueryResponse(BaseModel):
    """Generic query response"""

    query_type: str = Field(..., description="Type of query executed")
    parameters: QueryParameters = Field(..., description="Parameters used in the query")
    result_count: int = Field(..., description="Number of results returned")
    results: List[
        Union[
            Person, Company, Employment, PersonConnection, CompanyStat, Dict[str, Any]
        ]
    ] = Field(..., description="Query results")


class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(..., description="Error message")
