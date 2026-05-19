import time
import pytest
import jwt
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.services.api_gateway.src.middleware.auth import JWTAuthMiddleware
from backend.services.api_gateway.src.config import Settings

# 1. Generate real RS256 key pair
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)
public_key = private_key.public_key()

private_key_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

public_key_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

@pytest.fixture
def app():
    settings = Settings(
        KEYCLOAK_JWKS_URL="http://mock-keycloak/certs",
        KEYCLOAK_AUDIENCE="factory-ai-brain"
    )
    test_app = FastAPI()
    
    test_app.add_middleware(JWTAuthMiddleware, settings=settings)
    
    @test_app.get("/protected")
    async def protected_route(request: Request):
        return {
            "tenant_id": request.state.tenant_id,
            "user_id": request.state.user_id,
            "roles": request.state.roles
        }
        
    return test_app

@pytest.fixture
def client(app):
    return TestClient(app)

def create_token(payload: dict, expires_in: int = 3600, key=private_key_pem) -> str:
    now = int(time.time())
    base_payload = {
        "iat": now,
        "exp": now + expires_in,
        "aud": "account",
        "sub": "user-1",
        "tenant_id": "test-tenant"
    }
    base_payload.update(payload)
    return jwt.encode(base_payload, key, algorithm="RS256")


@patch("backend.services.api_gateway.src.middleware.auth.PyJWKClient.get_signing_key_from_jwt")
def test_valid_token(mock_get_key, client):
    # 2. Sign a JWT
    token = create_token({})
    
    # 3. Mock PyJWKClient
    mock_key = MagicMock()
    mock_key.key = public_key_pem
    mock_get_key.return_value = mock_key

    # 4. Verify request state
    response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "test-tenant"
    assert response.json()["user_id"] == "user-1"


@patch("backend.services.api_gateway.src.middleware.auth.PyJWKClient.get_signing_key_from_jwt")
def test_forged_token(mock_get_key, client):
    # 5. Verify forged token returns 401
    forged_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged_pem = forged_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    forged_token = create_token({}, key=forged_pem)

    mock_key = MagicMock()
    mock_key.key = public_key_pem # Middleware uses the real public key
    mock_get_key.return_value = mock_key

    response = client.get("/protected", headers={"Authorization": f"Bearer {forged_token}"})
    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"
    assert response.json()["message"] == "Invalid token"


@patch("backend.services.api_gateway.src.middleware.auth.PyJWKClient.get_signing_key_from_jwt")
def test_expired_token(mock_get_key, client):
    # 6. Verify expired token returns 401
    expired_token = create_token({}, expires_in=-3600)
    
    mock_key = MagicMock()
    mock_key.key = public_key_pem
    mock_get_key.return_value = mock_key

    response = client.get("/protected", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401
    assert response.json()["error_code"] == "UNAUTHORIZED"
    assert response.json()["message"] == "Token has expired"
