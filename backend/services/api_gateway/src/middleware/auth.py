import logging
import time
from typing import Callable, Optional

import jwt
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from jwt import PyJWKClient, PyJWKClientError
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import Settings

logger = logging.getLogger(__name__)

SKIP_PATHS = {"/health", "/api/docs", "/api/openapi.json", "/openapi.json", "/metrics"}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates RS256 JWTs issued by Keycloak.
    JWKS are fetched from Keycloak on first request and cached in-process.
    PyJWKClient automatically refreshes the cache when a kid is unknown.
    """

    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._jwks_client = PyJWKClient(
            settings.KEYCLOAK_JWKS_URL,
            cache_keys=True,
            lifespan=3600,          # Re-fetch JWKS every hour
        )
        self._audience = "account"
        self._algorithms = ["RS256"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in SKIP_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return self._unauthorized(request, "Missing or malformed Authorization header")

        token = auth_header.removeprefix("Bearer ").strip()

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload: dict = jwt.decode(
                token,
                signing_key.key,
                algorithms=self._algorithms,
                audience=self._audience,
                options={"require": ["exp", "sub", "iat"]},
            )
        except PyJWKClientError as exc:
            logger.warning("JWKS fetch/key-match failure: %s", exc)
            return self._unauthorized(request, "Unable to verify token signature")
        except jwt.ExpiredSignatureError:
            return self._unauthorized(request, "Token has expired")
        except jwt.InvalidTokenError as exc:
            logger.info("Invalid JWT from %s: %s", request.client.host if request.client else "unknown", exc)
            return self._unauthorized(request, "Invalid token")

        # Inject verified claims into request state
        request.state.user_id    = payload.get("sub")
        request.state.tenant_id  = payload.get("tenant_id") or payload.get("azp")
        request.state.factory_id = payload.get("factory_id")
        request.state.roles      = (
            payload.get("realm_access", {}).get("roles", [])
            + payload.get("resource_access", {}).get("factory-ai-brain", {}).get("roles", [])
        )

        if not request.state.tenant_id:
            logger.error("JWT missing tenant_id claim. sub=%s", request.state.user_id)
            return self._unauthorized(request, "Token missing required tenant_id claim")

        return await call_next(request)

    @staticmethod
    def _unauthorized(request: Request, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content={
                "error_code": "UNAUTHORIZED",
                "message": message,
                "request_id": getattr(request.state, "trace_id", None),
                "timestamp": time.time(),
            },
        )

# Example usage in app.py:
# from backend.services.api_gateway.src.config import get_settings
# settings = get_settings()
# app.add_middleware(JWTAuthMiddleware, settings=settings)
