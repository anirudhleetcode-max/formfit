import logging
import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("formfit.config")

PLACEHOLDER_SECRETS = {"", "change-me-in-production", "replace-with-a-long-random-string", "secret", "changeme"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "FormFit"
    env: str = "development"            # development | production
    log_level: str = "INFO"
    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_db: str = "formfit"
    jwt_secret: str = ""
    jwt_expire_minutes: int = 60 * 24 * 7
    model_path: str = "ml/artifacts/form_model.joblib"
    max_reps_per_session: int = 500
    max_body_bytes: int = 1_000_000
    login_max_failures: int = 10        # per email + client IP ...
    login_window_seconds: int = 900     # ... within this window
    stale_session_hours: int = 6        # active sessions older than this are marked abandoned
    cors_origins: str = "http://localhost:5175,http://127.0.0.1:5175"


def _check_secret(s: Settings) -> Settings:
    weak = s.jwt_secret in PLACEHOLDER_SECRETS or len(s.jwt_secret) < 32
    if not weak:
        return s
    if s.env.lower() != "development":
        raise RuntimeError(
            "JWT_SECRET is missing or a placeholder. Generate one with "
            "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"` and set it in backend/.env")
    log.warning("JWT_SECRET is missing/weak; using an ephemeral random secret (ENV=development). "
                "Tokens will stop working when the server restarts.")
    s.jwt_secret = secrets.token_urlsafe(48)
    return s


@lru_cache
def get_settings() -> Settings:
    return _check_secret(Settings())
