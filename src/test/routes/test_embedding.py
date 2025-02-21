import numpy as np
import pytest
from httpx import AsyncClient

from src.models.embedding import Keywords


@pytest.mark.anyio
async def test_create_embeddings_success(async_client: AsyncClient, auth_headers: dict):
    """Test successful creation of embeddings."""
    test_keywords: Keywords = Keywords(keywords=["test", "example", "word"])
    response = await async_client.post(
        "/v1/embedding/keywords",
        json=test_keywords.model_dump(),
        headers=auth_headers,  # Add this if you want to keep authentication
    )

    assert response.status_code == 201
    data = response.json()

    assert isinstance(data, dict)
    assert "keywords" in data
    assert len(data["keywords"]) == len(test_keywords.keywords)

    for keyword in data["keywords"]:
        assert "word" in keyword
        assert "x" in keyword
        assert "y" in keyword
        assert isinstance(keyword["x"], float)
        assert isinstance(keyword["y"], float)


@pytest.mark.anyio
async def test_create_embeddings_empty_list(
    async_client: AsyncClient, auth_headers: dict
):
    """Test creating embeddings with empty keyword list."""
    test_keywords: Keywords = Keywords(keywords=[])
    response = await async_client.post(
        "/v1/embedding/keywords", json=test_keywords.model_dump(), headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()
    assert data["keywords"] == []


@pytest.mark.anyio
async def test_create_embeddings_invalid_input(
    async_client: AsyncClient, auth_headers: dict
):
    """Test creating embeddings with invalid input."""
    invalid_data = {"invalid_field": ["test"]}
    response = await async_client.post(
        "/v1/embedding/keywords", json=invalid_data, headers=auth_headers
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.anyio
async def test_create_embeddings_single_keyword(
    async_client: AsyncClient, auth_headers: dict
):
    """Test creating embeddings with a single keyword."""
    test_keywords: Keywords = Keywords(keywords=["test"])
    response = await async_client.post(
        "/v1/embedding/keywords", json=test_keywords.model_dump(), headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()
    assert len(data["keywords"]) == 1
    assert data["keywords"][0]["word"] == "test"


@pytest.mark.anyio
async def test_create_embeddings_long_text(
    async_client: AsyncClient, auth_headers: dict
):
    """Test creating embeddings with longer text inputs."""
    test_keywords: Keywords = Keywords(
        keywords=[
            "this is a much longer piece of text",
            "another long sentence with multiple words",
        ]
    )
    response = await async_client.post(
        "/v1/embedding/keywords", json=test_keywords.model_dump(), headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()
    assert len(data["keywords"]) == 2


@pytest.mark.anyio
async def test_create_embeddings_special_characters(
    async_client: AsyncClient, auth_headers: dict
):
    """Test creating embeddings with special characters."""
    test_keywords: Keywords = Keywords(keywords=["test@#$%", "hello!", "world?"])
    response = await async_client.post(
        "/v1/embedding/keywords", json=test_keywords.model_dump(), headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()
    assert len(data["keywords"]) == 3


@pytest.mark.anyio
async def test_create_embeddings_invalid_types(
    async_client: AsyncClient, auth_headers: dict
):
    """Test creating embeddings with invalid types in the keywords list."""
    test_keywords = {"keywords": [123, True, "test"]}
    response = await async_client.post(
        "/v1/embedding/keywords", json=test_keywords, headers=auth_headers
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.anyio
async def test_create_embeddings_duplicates(
    async_client: AsyncClient, auth_headers: dict
):
    """Test creating embeddings with duplicate keywords."""
    test_keywords: Keywords = Keywords(keywords=["test", "test", "test"])
    response = await async_client.post(
        "/v1/embedding/keywords", json=test_keywords.model_dump(), headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()
    assert len(data["keywords"]) == 3

    # Compare coordinates using numpy's allclose for floating-point comparison
    coords = [(k["x"], k["y"]) for k in data["keywords"]]
    for coord in coords[1:]:
        assert np.allclose(coord, coords[0], rtol=1e-10, atol=1e-10)
