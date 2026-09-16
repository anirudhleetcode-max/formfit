from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "FormFit"
    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_db: str = "formfit"
    jwt_secret: str = "change-me-in-production"
    jwt_expire_minutes: int = 60 * 24 * 7
    model_path: str = "ml/artifacts/form_model.joblib"
    max_reps_per_session: int = 500
    cors_origins: str = "http://localhost:5175,http://127.0.0.1:5175"


@lru_cache
def get_settings() -> Settings:
    return Settings()
