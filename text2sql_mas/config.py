"""Pipeline configuration using pydantic-settings with environment variable binding."""

from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    groq_api_key: str                          # Set via GROQ_API_KEY env var
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.0              # Zero-temperature: deterministic
    groq_max_tokens: int = 1024
    groq_timeout_seconds: int = 30

    embedding_model: str = "all-MiniLM-L6-v2"
    schema_top_k: int = 5
    fk_closure_hops: int = 2
    max_correction_retries: int = 3

    db_dialect: str = "sqlite"                 # "sqlite" | "postgresql"
    db_path: str = "./databases/"

    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "text2sql_mas_v1"

    class Config:
        env_file = ".env"


config = PipelineConfig()
