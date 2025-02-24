from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket

from src.security.rateLimiter import FastAPILimiter
from src.security.rateLimiter.depends import RateLimiter, WebSocketRateLimiter


# Test Rate Limiter Initialization
@pytest.mark.asyncio
async def test_fastapi_limiter_init(mocker):
    redis_mock = mocker.AsyncMock()
    redis_mock.script_load.return_value = "test_sha"
    await FastAPILimiter.init(redis_mock)
    assert FastAPILimiter.redis == redis_mock
    assert FastAPILimiter.lua_sha == "test_sha"
    redis_mock.script_load.assert_called_once_with(FastAPILimiter.lua_script)
    await FastAPILimiter.close()


# Test Rate Limiter Dependency
@pytest.mark.asyncio
async def test_rate_limiter_under_limit():
    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route(rate_limit: None = Depends(RateLimiter(times=2, seconds=5))):
        return {"status": "ok"}

    # Initialize rate limiter with dummy backend
    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0, 0]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    with TestClient(test_app) as client:
        response = client.get("/test")
        assert response.status_code == 200

        # Second request within limit
        response = client.get("/test")
        assert response.status_code == 200

    await FastAPILimiter.close()


@pytest.mark.asyncio
async def test_rate_limiter_exceeds_limit():
    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route(rate_limit: None = Depends(RateLimiter(times=1, seconds=5))):
        return {"status": "ok"}

    # Initialize rate limiter with dummy backend
    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0, 1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    with TestClient(test_app) as client:
        response = client.get("/test")
        assert response.status_code == 200

        # Second request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    await FastAPILimiter.close()


# Test WebSocket Rate Limiter
@pytest.mark.asyncio
async def test_websocket_rate_limiter():
    # Create a more complete WebSocket mock
    ws_mock = AsyncMock(spec=WebSocket)
    ws_mock.scope = {"path": "/test"}
    ws_mock.client = AsyncMock()
    ws_mock.client.host = "127.0.0.1"
    ws_mock.headers = {}

    limiter = WebSocketRateLimiter(times=1, seconds=5)

    # Initialize rate limiter with dummy backend
    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0, 1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    # First connection should succeed
    await limiter(ws_mock)
    assert not ws_mock.close.called

    # Second connection should be rate limited
    await limiter(ws_mock)
    ws_mock.close.assert_called_once_with(code=1013)

    await FastAPILimiter.close()


# Test Error Cases
@pytest.mark.asyncio
async def test_rate_limiter_without_init():
    FastAPILimiter.redis = None
    limiter = RateLimiter(times=1, seconds=5)

    request_mock = AsyncMock()
    response_mock = AsyncMock()

    with pytest.raises(Exception, match="You must call FastAPILimiter.init"):
        await limiter(request_mock, response_mock)


@pytest.mark.asyncio
async def test_rate_limiter_with_invalid_identifier():
    async def invalid_identifier(request):
        raise ValueError("Invalid request")

    # Initialize rate limiter with dummy backend
    redis_mock = AsyncMock()
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    limiter = RateLimiter(times=1, seconds=5, identifier=invalid_identifier)
    request_mock = AsyncMock()
    response_mock = AsyncMock()

    with pytest.raises(Exception, match="Error computing rate key"):
        await limiter(request_mock, response_mock)

    await FastAPILimiter.close()


# Test Custom Callbacks
@pytest.mark.asyncio
async def test_rate_limiter_custom_callback():
    async def custom_callback(request, response, pexpire):
        response.status_code = 418  # I'm a teapot
        response.headers["X-RateLimit-Retry-After"] = str(pexpire)
        return response  # Return the response directly

    test_app = FastAPI()

    @test_app.get("/test")
    async def test_route(
        rate_limit: None = Depends(
            RateLimiter(times=1, seconds=5, callback=custom_callback)
        ),
    ):
        return {"status": "ok"}

    # Initialize rate limiter with dummy backend
    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0, 1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    with TestClient(test_app) as client:
        response = client.get("/test")
        assert response.status_code == 200

        # Second request should use custom callback
        response = client.get("/test")
        assert response.status_code == 418
        assert "X-RateLimit-Retry-After" in response.headers

    await FastAPILimiter.close()


# Additional Integration Tests
@pytest.mark.asyncio
async def test_multiple_rate_limiters():
    test_app = FastAPI()

    @test_app.get("/multi-limit")
    async def test_route(
        rate_limit_1: None = Depends(RateLimiter(times=2, seconds=5)),
        rate_limit_2: None = Depends(RateLimiter(times=1, minutes=1)),
    ):
        return {"status": "ok"}

    redis_mock = AsyncMock()
    # First limiter allows 2 requests, second only 1
    redis_mock.evalsha.side_effect = [0, 0, 1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    with TestClient(test_app) as client:
        # First request passes both limiters
        response = client.get("/multi-limit")
        assert response.status_code == 200

        # Second request blocked by second limiter
        response = client.get("/multi-limit")
        assert response.status_code == 429

    await FastAPILimiter.close()


@pytest.mark.asyncio
async def test_different_time_windows():
    test_app = FastAPI()
    times = 3

    @test_app.get("/sliding-window")
    async def test_route(
        rate_limit: None = Depends(RateLimiter(times=times, milliseconds=500)),
    ):
        return {"status": "ok"}

    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0] * times + [1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    with TestClient(test_app) as client:
        # Make exactly 'times' requests
        for _ in range(times):
            response = client.get("/sliding-window")
            assert response.status_code == 200

        # Next request should be rate limited
        response = client.get("/sliding-window")
        assert response.status_code == 429

        # Verify the retry-after header is close to 500ms
        retry_after = int(response.headers["Retry-After"])
        assert 0 < retry_after <= 1

    await FastAPILimiter.close()


@pytest.mark.asyncio
async def test_path_specific_rate_limiting():
    test_app = FastAPI()

    @test_app.get("/path1")
    async def route1(rate_limit: None = Depends(RateLimiter(times=1, seconds=5))):
        return {"path": 1}

    @test_app.get("/path2")
    async def route2(rate_limit: None = Depends(RateLimiter(times=1, seconds=5))):
        return {"path": 2}

    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0, 0, 1000, 1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    with TestClient(test_app) as client:
        # First requests to both paths should succeed
        assert client.get("/path1").status_code == 200
        assert client.get("/path2").status_code == 200

        # Second requests to both paths should be limited
        assert client.get("/path1").status_code == 429
        assert client.get("/path2").status_code == 429

    await FastAPILimiter.close()


@pytest.mark.asyncio
async def test_ip_based_rate_limiting():
    test_app = FastAPI()

    @test_app.get("/ip-test")
    async def test_route(rate_limit: None = Depends(RateLimiter(times=1, seconds=5))):
        return {"status": "ok"}

    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0, 0, 1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    with TestClient(test_app) as client:
        # Request with direct IP
        response = client.get("/ip-test")
        assert response.status_code == 200

        # Request with different forwarded IP should get its own limit
        response = client.get("/ip-test", headers={"X-Forwarded-For": "1.2.3.4"})
        assert response.status_code == 200

        # Second request with same forwarded IP should be limited
        response = client.get("/ip-test", headers={"X-Forwarded-For": "1.2.3.4"})
        assert response.status_code == 429

    await FastAPILimiter.close()


@pytest.mark.asyncio
async def test_websocket_context_rate_limiting():
    # Create WebSocket mocks with different contexts
    ws1 = AsyncMock(spec=WebSocket)
    ws1.scope = {"path": "/ws"}
    ws1.client = AsyncMock(host="127.0.0.1")
    ws1.headers = {}

    ws2 = AsyncMock(spec=WebSocket)
    ws2.scope = {"path": "/ws"}
    ws2.client = AsyncMock(host="127.0.0.1")
    ws2.headers = {}

    limiter = WebSocketRateLimiter(times=1, seconds=5)

    redis_mock = AsyncMock()
    redis_mock.evalsha.side_effect = [0, 0, 1000]
    redis_mock.script_load.return_value = "dummy_sha"
    await FastAPILimiter.init(redis_mock)

    # First connection with context "user1" succeeds
    await limiter(ws1, context_key="user1")
    assert not ws1.close.called

    # Connection with different context "user2" succeeds
    await limiter(ws2, context_key="user2")
    assert not ws2.close.called

    # Second connection with context "user1" is limited
    await limiter(ws1, context_key="user1")
    ws1.close.assert_called_once_with(code=1013)

    await FastAPILimiter.close()


@pytest.mark.asyncio
async def test_rate_limiter_cleanup():
    """Test that FastAPILimiter.close() properly cleans up resources"""
    redis_mock = AsyncMock()
    await FastAPILimiter.init(redis_mock)

    # Verify initialization
    assert FastAPILimiter.redis is not None

    # Close and verify cleanup
    await FastAPILimiter.close()
    assert FastAPILimiter.redis is None
