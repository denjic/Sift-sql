# Multi-Agent Text-to-SQL System

> A stateful, multi-agent pipeline that converts natural language questions into accurate SQLite queries — powered by **LangGraph**, **Groq (LLaMA 3.3 70B)**, and **sentence-transformers**.

---

## Overview

Most Text-to-SQL systems hand a raw question and an entire database schema to a single LLM and hope for the best. This project takes a fundamentally different approach: it decomposes the task across **four specialized agents** orchestrated by a **LangGraph StateGraph**, mirroring how a human engineering team would tackle the problem — a data retriever, a database architect, a code developer, and a QA tester.

The result is a system that is dramatically more reliable on complex, multi-table queries, handles schema hallucinations proactively, and self-corrects its own mistakes before ever touching the database.

---

## Architecture

```
Natural Language Query
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Phase 1 — Semantic Schema Linking Agent                  │
│  • Encodes all tables with all-MiniLM-L6-v2               │
│  • Cosine similarity → Top-K table retrieval              │
│  • Foreign key graph closure (up to 2 hops)               │
│  • Output: Pruned schema (4-6 tables vs. potentially 100+)│
└────────────────────────┬──────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────┐
│  Phase 2 — Chain-of-Thought Query Planning Agent          │
│  • Reads pruned schema + question                         │
│  • Outputs a structured 5-field logical blueprint         │
│    (Strategy, Tables, Joins, Aggregation, Special)        │
│  • Strictly prohibited from writing any SQL               │
└────────────────────────┬──────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────┐
│  Phase 3 — Constrained SQL Generation Agent               │
│  • Groq API: LLaMA 3.3 70B @ Temperature = 0.0            │
│  • Translates blueprint → raw SQLite syntax               │
│  • Supports INTERSECT, EXCEPT, UNION, subqueries, HAVING  │
└────────────────────────┬──────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────┐
│  Phase 4 — AST Schema Guard + Self-Correction Loop        │
│  • sqlglot parses SQL into an Abstract Syntax Tree        │
│  • Validates every table and column against pruned schema │
│  • On failure → Self-Correction Agent rewrites the query  │
│  • Retries up to N=3 times before surfacing the error     │
└────────────────────────┬──────────────────────────────────┘
                         │
                         ▼
               Executed SQL Result
```

---

## Key Features

| Feature | Description |
|---|---|
| **Semantic Schema Linking** | Uses `all-MiniLM-L6-v2` embeddings + cosine similarity to prune schemas before any LLM call, eliminating the "lost-in-the-middle" hallucination problem |
| **FK Graph Closure** | Automatically discovers and includes required bridge/join tables via foreign key traversal (up to 2 hops), ensuring join paths are never accidentally severed |
| **CoT Planning** | Separates architectural reasoning from code generation — the planner produces a structured blueprint; the generator only implements it |
| **AST Validation** | `sqlglot` intercepts malformed SQL *before* it reaches the database, providing precise, structured error messages for the correction agent |
| **Self-Correction Loop** | LangGraph cycles the corrector back through validation up to 3 times, achieving exponential reliability improvement over linear systems |
| **Deterministic LLM Calls** | All Groq API calls use `temperature=0.0` for reproducible, debuggable outputs |
| **Benchmark Evaluation** | Built-in Spider dataset evaluation with execution accuracy (EX) scoring, multi-config A/B testing, checkpoint/resume support, and Excel/CSV export |
| **Streamlit UI** | Interactive demo + full evaluation dashboard with real-time trace visualisation |

---

## Agents in Detail

### Agent 1 — Semantic Schema Linking (Deterministic)
- Serialises the database into compact strings (`table: col1(type), col2(type)`)
- Embeds them locally with `all-MiniLM-L6-v2` (22M params, no API call needed)
- Retrieves the Top-K most semantically similar tables via cosine similarity
- Runs a FK closure pass to preserve join integrity

### Agent 2 — Chain-of-Thought Planner
- Outputs a strict 5-field blueprint: `Strategy | Tables | Joins | Aggregation | Special`
- Explicitly forbidden from writing SQL — pure architectural reasoning only
- Handles complex query types: `INTERSECT`, `EXCEPT`, `UNION`, `GROUP BY`, `HAVING`, subqueries

### Agent 3 — SQL Generator
- Consumes the blueprint and pruned schema
- Uses `LLaMA-3.3-70b-versatile` via Groq at `temperature=0.0`
- Constrained by a highly structured system prompt with few-shot SQL examples
- Supports full SQL complexity: set operations, aggregations, nested subqueries

### Agent 4 — AST Guard + Self-Corrector
- Passes the generated SQL through `sqlglot`'s AST parser
- Traverses every `Table` and `Column` node, checking against the pruned schema dictionary
- On mismatch → generates a structured error (e.g., *"Column 'employee_count' not in 'regions'"*) and passes it to the corrector LLM
- Cycles back via LangGraph until validation passes or the retry ceiling is hit

---

## Technology Stack

| Component | Technology | Reason |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Enables typed state passing, conditional edges, and cyclic self-correction loops |
| LLM | Groq — LLaMA 3.3 70B Versatile | Fast inference, strong SQL reasoning, cost-effective |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Lightweight (22M params), CPU-fast, no API dependency |
| SQL Parsing | `sqlglot` | Multi-dialect AST parsing without requiring a live DB connection |
| Database / Benchmark | SQLite + Spider Dataset | Lightweight, no Docker; Spider provides complex multi-domain schemas for rigorous evaluation |
| Frontend | Streamlit | Rapid Python-native UI with real-time trace logging |
| Data Processing | pandas, plotly | Evaluation metrics, charts, Excel export |

---

## Benchmark Evaluation Configurations

The system supports four evaluation modes for ablation studies:

| Config | Schema | Planning | Notes |
|---|---|---|---|
| `baseline` | Full raw DDL | None | Zero-shot monolithic LLM — the control group |
| `model_a` | Pruned (MiniLM) | None | Tests the isolated effect of schema linking |
| `model_b` | Pruned (MiniLM) | None + expert rules | Schema linking + constrained generation prompt |
| `model_c` | Pruned (MiniLM) | CoT Blueprint | Full multi-agent pipeline — all four phases |

Results are scored with **Execution Accuracy (EX)**: the generated SQL and the gold SQL must return identical result sets.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/)
- The [Spider dataset](https://yale-lily.github.io/spider) (place in `./spider/`)

### Installation

```bash
git clone https://github.com/your-username/Multi_Agent-TexttoSql.git
cd Multi_Agent-TexttoSql
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
streamlit run app.py
```

---

## Project Structure

```
Multi_Agent-TexttoSql/
├── app.py                      # Main Streamlit app — full pipeline + evaluation dashboard
├── SYSTEM_ARCHITECTURE_DOC.md  # Detailed architectural documentation
├── requirements.txt            # Python dependencies
├── analyze_ast.py              # AST analysis utility
├── analyze_eval.py             # Evaluation result analysis
├── compare_errors.py           # Error comparison across configurations
├── compare_traces.py           # Trace comparison utility
├── spider/                     # Spider benchmark dataset (dev.json + databases/)
├── databases/                  # SQLite database files
├── checkpoints/                # Evaluation checkpoint files (auto-generated)
└── logs/                       # Evaluation trace logs (JSONL + Excel)
```

---

## Design Philosophy

**Why multi-agent instead of a single large prompt?**

Single-LLM approaches suffer from three compounding failure modes at scale:
1. **Context overflow** — real databases have hundreds of tables; injecting all of them degrades attention quality
2. **Concurrent reasoning failure** — asking a model to simultaneously understand schema, plan joins, and format syntax increases hallucination rates sharply
3. **Silent failure** — if the SQL is wrong, a monolithic system has no mechanism to self-diagnose or repair

This system resolves all three: schema linking caps context, planning separates concerns, and AST validation + self-correction provides a structured safety net.

---

## Future Scope

- **Clarification Agent** — catch ambiguous questions pre-planning and prompt the user for intent
- **Dynamic Schema Discovery** — live `INFORMATION_SCHEMA` sweeps for production databases
- **NoSQL / Federated DB Support** — extend the AST layer to target MongoDB or other dialects via `sqlglot`'s multi-dialect support
- **RAG-based Value Mapping** — index column *values* to match entity mentions (e.g., "Acme Corp") to the correct column without hallucination

---

## Acknowledgements

- [Spider Dataset](https://yale-lily.github.io/spider) — Yale Semantic Parsing and Text-to-SQL benchmark
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful multi-agent orchestration
- [Groq](https://groq.com/) — ultra-fast LLM inference
- [sqlglot](https://github.com/tobymao/sqlglot) — Python SQL parser and transpiler
- [sentence-transformers](https://www.sbert.net/) — local dense embeddings

---

## License

This project is for academic and research purposes.
