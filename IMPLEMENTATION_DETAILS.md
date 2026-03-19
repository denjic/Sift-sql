# IMPLEMENTATION_DETAILS.md
# Implementation Details: Technology Stack, Configuration, and Deployment

---

## 1. Technology Stack Overview

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Orchestration | LangGraph | ≥0.1.0 | Stateful agent graph; typed state; conditional edges; loop support |
| Language | Python | ≥3.11 | Type hint support; async compatibility; rich ML ecosystem |
| Schema Embeddings | sentence-transformers | ≥2.7.0 | `all-MiniLM-L6-v2` for fast, high-quality semantic similarity |
| SQL Parsing & Validation | sqlglot | ≥23.0 | Cross-dialect AST parsing; zero DB dependency for validation |
| LLM Inference | Groq API | latest | Sub-200ms inference latency; `qwen3-32b` hosted endpoint |
| Database (dev/test) | SQLite | 3.x | Spider/BIRD databases ship as SQLite; zero-config testing |
| Database (production) | PostgreSQL | ≥14 | Enterprise-grade; extended `INFORMATION_SCHEMA` support |
| HTTP Client | httpx / requests | ≥0.27.0 | Groq API calls |
| Configuration | pydantic-settings | ≥2.0 | Typed configuration management; environment variable binding |
| Testing | pytest | ≥8.0 | Unit and integration tests for each agent node |
| Experiment Tracking | MLflow | ≥2.12 | Logging EX/EM per query; ablation run management |

---

## 2. Project Structure

```
text2sql_mas/
├── agents/
│   ├── __init__.py
│   ├── schema_linker.py        # Phase 1: MiniLM cosine retrieval
│   ├── query_planner.py        # Phase 2: CoT blueprint generation
│   ├── sql_generator.py        # Phase 3: Constrained SQL generation
│   ├── ast_guard.py            # Phase 4A: AST schema validation
│   └── self_corrector.py       # Phase 4B: Iterative repair
├── graph/
│   ├── __init__.py
│   ├── state.py                # SQLAgentState TypedDict definition
│   ├── builder.py              # LangGraph StateGraph construction
│   └── router.py               # Conditional edge routing functions
├── schema/
│   ├── __init__.py
│   ├── serializer.py           # Table-to-string serialization
│   ├── embedder.py             # MiniLM embedding + cosine retrieval
│   ├── fk_closure.py           # Foreign key closure algorithm
│   └── formatter.py            # Pruned schema → prompt string
├── llm/
│   ├── __init__.py
│   ├── groq_client.py          # Groq API wrapper with retry logic
│   └── prompts.py              # All system prompt templates
├── evaluation/
│   ├── __init__.py
│   ├── spider_loader.py        # Spider dataset loader
│   ├── bird_loader.py          # BIRD dataset loader
│   ├── metrics.py              # EX, EM, VES computation
│   └── ablation.py             # Ablation study runner
├── config.py                   # pydantic-settings configuration
├── main.py                     # Pipeline entry point
├── run_experiments.py          # Experiment runner (Spider/BIRD)
└── tests/
    ├── test_schema_linker.py
    ├── test_ast_guard.py
    ├── test_self_corrector.py
    └── test_full_pipeline.py
```

---

## 3. LangGraph StateGraph Construction

```python
from langgraph.graph import StateGraph, END
from graph.state import SQLAgentState
from agents.schema_linker import schema_linking_node
from agents.query_planner import query_planning_node
from agents.sql_generator import sql_generation_node
from agents.ast_guard import ast_guard_node
from agents.self_corrector import self_correction_node
from graph.router import route_after_guard

def build_pipeline(max_corrections: int = 3) -> callable:
    """Constructs and compiles the four-phase LangGraph pipeline."""

    graph = StateGraph(SQLAgentState)

    # Register agent nodes
    graph.add_node("schema_linker", schema_linking_node)
    graph.add_node("query_planner", query_planning_node)
    graph.add_node("sql_generator", sql_generation_node)
    graph.add_node("ast_guard", ast_guard_node)
    graph.add_node("self_corrector", self_correction_node)

    # Sequential edges: linker → planner → generator → guard
    graph.set_entry_point("schema_linker")
    graph.add_edge("schema_linker", "query_planner")
    graph.add_edge("query_planner", "sql_generator")
    graph.add_edge("sql_generator", "ast_guard")

    # Conditional edge: guard → END (pass) or self_corrector (fail)
    graph.add_conditional_edges(
        "ast_guard",
        route_after_guard,
        {
            "validated": END,
            "correct": "self_corrector",
            "exhausted": END  # Pipeline failed; returns best-attempt SQL
        }
    )

    # Correction loop: self_corrector feeds back into ast_guard
    graph.add_edge("self_corrector", "ast_guard")

    return graph.compile()
```

---

## 4. Groq API Client Configuration

```python
# config.py
from pydantic_settings import BaseSettings

class PipelineConfig(BaseSettings):
    groq_api_key: str                          # Set via GROQ_API_KEY env var
    groq_model: str = "qwen3-32b"
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
```

```python
# llm/groq_client.py
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class GroqClient:
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.groq_api_key}",
            "Content-Type": "application/json"
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": self.config.groq_model,
            "temperature": self.config.groq_temperature,
            "max_tokens": self.config.groq_max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }
        with httpx.Client(timeout=self.config.groq_timeout_seconds) as client:
            response = client.post(self.BASE_URL, json=payload, headers=self.headers)
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
```

**Retry policy:** The Groq API is subject to rate limiting under high concurrency. The `tenacity` retry decorator applies exponential backoff (1s → 2s → 4s) up to 3 attempts before raising, avoiding pipeline failures from transient network errors.

---

## 5. Database Connectors

### 5.1 SQLite (Development and Benchmark Evaluation)

```python
import sqlite3

def execute_query(sql: str, db_path: str) -> tuple[list, list[str]]:
    """
    Executes a SQL query against a SQLite database.
    Returns (rows, column_names). Used for EX evaluation only.
    AST validation fires before this call in the pipeline.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        return [tuple(row) for row in rows], col_names
    finally:
        conn.close()
```

### 5.2 Schema Loading from SQLite (Spider Format)

```python
def load_schema_from_sqlite(db_path: str) -> dict:
    """
    Introspects a SQLite database to extract the full schema.
    Parses table names, column names, types, and FK relationships.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {}
    for tbl in tables:
        cursor.execute(f"PRAGMA table_info('{tbl}')")
        cols = [
            {"name": row[1], "type": row[2], "is_pk": bool(row[5])}
            for row in cursor.fetchall()
        ]
        cursor.execute(f"PRAGMA foreign_key_list('{tbl}')")
        for fk in cursor.fetchall():
            # fk[3] = from_col, fk[2] = ref_table, fk[4] = ref_col
            for col in cols:
                if col["name"] == fk[3]:
                    col["fk_ref"] = f"{fk[2]}.{fk[4]}"
        schema[tbl] = {"columns": cols}

    conn.close()
    return schema
```

### 5.3 PostgreSQL (Production)

```python
import psycopg2

def load_schema_from_postgres(connection_string: str) -> dict:
    """Loads schema from PostgreSQL INFORMATION_SCHEMA."""
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT table_name, column_name, data_type,
               (SELECT constraint_type FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                WHERE kcu.table_name = c.table_name
                AND kcu.column_name = c.column_name
                AND constraint_type = 'PRIMARY KEY' LIMIT 1) as pk
        FROM information_schema.columns c
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    # ... parse results into schema dict
```

---

## 6. Prompt Template Management

All prompts are stored as versioned template strings in `llm/prompts.py`, enabling A/B testing of prompt variants:

```python
PLANNING_PROMPT_V1 = """
You are a database query planner. Given a natural language question and a pruned schema,
produce a structured logical blueprint. Do NOT generate SQL.

Schema:
{pruned_schema}

Question:
{nl_query}

Logical Blueprint (list: required tables, joins, filters, aggregations):
"""

GENERATION_PROMPT_V1 = """
You are a SQL generation expert. Generate valid SQLite SQL.
RULES: Use ONLY the tables and columns in the schema. Return ONLY the SQL, no explanation.

Schema:
{pruned_schema}

Blueprint:
{logical_blueprint}

Question:
{nl_query}

SQL:
"""

CORRECTION_PROMPT_V1 = """
The SQL query below has schema errors. Fix it using ONLY the columns listed in the schema.

Original SQL:
{generated_sql}

Errors detected:
{error_message}

Schema:
{pruned_schema}

Blueprint:
{logical_blueprint}

Corrected SQL:
"""
```

---

## 7. MLflow Experiment Tracking

```python
import mlflow

def run_evaluation_with_tracking(pipeline, dataset, experiment_name: str):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        results = []
        for sample in dataset:
            state = pipeline.invoke({
                "nl_query": sample["question"],
                "full_schema": sample["schema"],
                "db_path": sample["db_path"],
                "correction_attempts": 0,
                "max_corrections": config.max_correction_retries
            })
            results.append({
                "question": sample["question"],
                "predicted_sql": state["final_sql"],
                "gold_sql": sample["gold_sql"],
                "status": state["pipeline_status"],
                "corrections_used": state["correction_attempts"]
            })

        # Compute and log metrics
        ex_score = compute_execution_accuracy(results)
        em_score = compute_exact_match(results)
        correction_rate = sum(r["corrections_used"] > 0 for r in results) / len(results)

        mlflow.log_metric("execution_accuracy", ex_score)
        mlflow.log_metric("exact_match", em_score)
        mlflow.log_metric("correction_invocation_rate", correction_rate)
        mlflow.log_param("model", config.groq_model)
        mlflow.log_param("top_k", config.schema_top_k)
        mlflow.log_param("max_corrections", config.max_correction_retries)
```

---

## 8. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/[author]/text2sql-mas.git
cd text2sql-mas

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export GROQ_API_KEY="your_groq_api_key_here"

# 5. Download Spider dataset
wget https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGfBeCZum4q62kDO5 \
  -O spider.zip && unzip spider.zip

# 6. Run full pipeline on a single example
python main.py \
  --query "Find the names of all customers who placed orders after 2022" \
  --db "./spider/database/customers/customers.sqlite"

# 7. Run full Spider dev evaluation
python run_experiments.py \
  --dataset spider \
  --split dev \
  --config Model_C_full_agentic
```

---

## 9. Hardware Requirements

| Setting | Specification |
|---------|--------------|
| CPU (minimum) | 4-core, 8GB RAM |
| CPU (recommended) | 8-core, 16GB RAM (for parallel batch evaluation) |
| GPU | Not required (LLM inference offloaded to Groq) |
| Storage | ~5GB (Spider + BIRD databases + MiniLM model weights) |
| Internet | Required for Groq API calls (latency: ~150–300ms per call) |

The MiniLM embedding computation runs on CPU; the 22M-parameter model fits comfortably in 200MB of RAM. No GPU is required for the embedding component.
