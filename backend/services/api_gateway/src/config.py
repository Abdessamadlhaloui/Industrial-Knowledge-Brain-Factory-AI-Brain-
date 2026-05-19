from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Keycloak / JWT
    KEYCLOAK_JWKS_URL: str = "http://keycloak:8080/realms/factory/protocol/openid-connect/certs"
    KEYCLOAK_AUDIENCE: str = "factory-ai-brain"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # Downstream services
    AGENT_SERVICE_URL: str = "http://agent-service:8002"
    RAG_SERVICE_URL: str = "http://rag-service:8001"
    TELEMETRY_SERVICE_URL: str = "http://telemetry-service:8003"
    KG_SERVICE_URL: str = "http://kg-service:8004"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
