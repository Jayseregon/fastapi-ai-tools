# ----------------------------------------------------------------------
# Portions of this code are derived from fastapi-limiter
# (https://github.com/long2ice/fastapi-limiter.git) which is licensed
# under the Apache License, Version 2.0.
#
# Modifications Copyright (C) Jayseregon, 2025.
#
# This file is part of fastapi-ai-tools which is licensed under LGPL-3.0-or-later.
#
# Licensed under the Apache License, Version 2.0 (for fastapi-limiter derived parts)
# and under LGPL-3.0-or-later for the overall project.
# You may obtain a copy of the Apache License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------

from typing import Annotated, Callable, Optional

import redis as pyredis
from pydantic import Field
from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocket

# Use relative import to reference the local module.
from . import FastAPILimiter


class RateLimiter:
    def __init__(
        self,
        times: Annotated[int, Field(ge=0)] = 1,
        milliseconds: Annotated[int, Field(ge=-1)] = 0,
        seconds: Annotated[int, Field(ge=-1)] = 0,
        minutes: Annotated[int, Field(ge=-1)] = 0,
        hours: Annotated[int, Field(ge=-1)] = 0,
        identifier: Optional[Callable] = None,
        callback: Optional[Callable] = None,
    ):
        self.times = times
        self.milliseconds = (
            milliseconds + 1000 * seconds + 60000 * minutes + 3600000 * hours
        )
        self.identifier = identifier
        self.callback = callback

    async def _check(self, key):
        redis_instance = FastAPILimiter.redis
        pexpire = await redis_instance.evalsha(
            FastAPILimiter.lua_sha, 1, key, str(self.times), str(self.milliseconds)
        )
        return pexpire

    async def __call__(self, request: Request, response: Response):
        if not FastAPILimiter.redis:
            raise Exception(
                "You must call FastAPILimiter.init in startup event of fastapi!"
            )
        route_index = 0
        dep_index = 0
        found = False
        for i, route in enumerate(request.app.routes):
            if route.path == request.scope["path"] and request.method in route.methods:
                route_index = i
                for j, dependency in enumerate(route.dependencies):
                    if self is dependency.dependency:
                        dep_index = j
                        found = True
                        break
                if found:
                    break
        identifier = self.identifier or FastAPILimiter.identifier
        if identifier is None:
            raise Exception("Identifier function not configured")
        try:
            rate_key = await identifier(request)
        except Exception as e:
            # Log and optionally handle identifier errors
            raise Exception("Error computing rate key.") from e

        key = f"{FastAPILimiter.prefix}:{rate_key}:{route_index}:{dep_index}"
        try:
            pexpire = await self._check(key)
        except pyredis.exceptions.NoScriptError:
            FastAPILimiter.lua_sha = await FastAPILimiter.redis.script_load(
                FastAPILimiter.lua_script
            )
            pexpire = await self._check(key)
        except Exception as e:
            # Log unexpected exceptions during Redis script execution
            raise Exception("Rate limiter check failed") from e

        callback = self.callback or FastAPILimiter.http_callback
        if callback is None:
            raise Exception("HTTP callback function not configured")
        if pexpire != 0:
            return await callback(request, response, pexpire)


class WebSocketRateLimiter(RateLimiter):
    # Use type: ignore to override signature differences
    async def __call__(self, ws: WebSocket, context_key=""):  # type: ignore[override]
        if not FastAPILimiter.redis:
            raise Exception(
                "You must call FastAPILimiter.init in startup event of fastapi!"
            )
        try:
            identifier = self.identifier or FastAPILimiter.identifier
            if identifier is None:
                raise Exception("Identifier function not configured")
            rate_key = await identifier(ws)
        except Exception as e:
            raise Exception("Error computing rate key for websocket.") from e
        key = f"{FastAPILimiter.prefix}:ws:{rate_key}:{context_key}"
        try:
            pexpire = await self._check(key)
        except Exception as e:
            raise Exception("WebSocket rate limiter check failed") from e
        callback = self.callback or FastAPILimiter.ws_callback
        if callback is None:
            raise Exception("WebSocket callback function not configured")
        if pexpire != 0:
            return await callback(ws, pexpire)
