"""Pipeline configuration — pydantic-settings based.

FIX LOG (v2):
- fk_closure_hops reduced from 2 → 1.
  2-hop closure on databases like car_1 drags in the entire FK graph
  (cars_data → car_names → model_list → car_makers → all tables), causing
  Phase 3 to generate unnecessary multi-table JOINs for simple single-table
  queries. 1 hop only adds directly adjacent tables, which is sufficient
  for 95% of Spider queries while avoiding over-retrieval.
"""

from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    # Groq LLM
    groq_api_key:        str   = ""
    groq_model:          str   = "qwen3-32b"
    groq_temperature:    float = 0.0
    groq_max_tokens:     int   = 1024
    groq_timeout_seconds: int  = 30

    # Schema linking
    embedding_model:     str = "all-MiniLM-L6-v2"
    schema_top_k:        int = 5
    fk_closure_hops:     int = 1        # CHANGED: was 2, now 1

    # Self-correction
    max_correction_retries: int = 3

    # Database
    db_dialect: str = "sqlite"
    db_path:    str = "./databases/"

    # Experiment tracking
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name:     str = "text2sql_mas_v1"

    class Config:
        env_file = ".env"
        extra    = "ignore"


config = PipelineConfig()