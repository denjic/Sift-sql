"""Phase 4B: Self-Correction Agent — iterative repair using structured AST errors.

FIX LOG (v2):
- BUG 1 FIXED: Removed `user_message=f"Fix this SQL: {generated_sql}"` — the SQL
  was appearing twice in the full prompt context (once in CORRECTION_PROMPT_V1 body
  via {generated_sql}, once in the user turn), causing the model to echo the
  original broken SQL back unchanged instead of correcting it.
- BUG 2 FIXED: CORRECTION_PROMPT_V1 is now used as the USER message, not as the
  system prompt. The system prompt is now a short, stable role description. This
  matches how Groq's qwen3-32b processes instruction-following tasks.
- BUG 3 FIXED: Added SQL fence stripping as a dedicated helper — was previously
  unreliable with triple-backtick variants like ```sql\n and ```\n.
"""

import re
from text2sql_mas.llm.groq_client import GroqClient
from text2sql_mas.llm.prompts import CORRECTION_PROMPT_V1
from text2sql_mas.schema.formatter import format_pruned_schema
from text2sql_mas.agents.ast_guard import format_errors_for_correction
from text2sql_mas.config import config


CORRECTOR_SYSTEM = (
    "You are an expert SQL debugger. When given a broken SQL query and a list "
    "of schema violations, you output only the corrected SQL query with no "
    "explanation, no markdown formatting, and no code fences."
)


def _strip_sql_fences(text: str) -> str:
    """Remove Qwen3 thinking blocks and markdown code fences from LLM output."""
    # Strip Qwen3 thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # Remove ```sql ... ``` or ``` ... ```
    text = re.sub(r"^```(?:sql)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    # Remove any remaining leading/trailing whitespace
    return text.strip()


def self_correction_node(state: dict) -> dict:
    """
    LangGraph node: Phase 4B — Self-Correction Agent.

    Input state keys:  generated_sql, ast_errors, pruned_schema,
                       logical_blueprint, correction_attempts, max_corrections
    Output state keys: generated_sql (updated), correction_attempts (incremented),
                       pipeline_status, final_sql (if exhausted)
    """
    generated_sql      = state["generated_sql"]
    ast_errors         = state.get("ast_errors", [])
    pruned_schema      = state["pruned_schema"]
    logical_blueprint  = state.get("logical_blueprint", "")
    correction_attempts = state.get("correction_attempts", 0)
    max_corrections    = state.get("max_corrections", config.max_correction_retries)

    # Format pruned schema for the prompt
    pruned_schema_str = format_pruned_schema(
        list(pruned_schema.keys()), pruned_schema
    )

    # Format structured errors into a readable directive
    error_message = format_errors_for_correction(ast_errors)

    # Build the correction USER message (SQL appears here only — not in system prompt)
    user_message = CORRECTION_PROMPT_V1.format(
        generated_sql=generated_sql,
        error_message=error_message,
        pruned_schema=pruned_schema_str,
        logical_blueprint=logical_blueprint,
    )

    # Call Groq LLM — system prompt is a short stable role, user message has all context
    client = GroqClient(config)
    raw_output = client.complete(
        system_prompt=CORRECTOR_SYSTEM,
        user_message=user_message,
    )

    corrected_sql = _strip_sql_fences(raw_output)

    # Safety fallback: if corrector returned empty string, keep the original
    if not corrected_sql:
        corrected_sql = generated_sql

    new_attempts = correction_attempts + 1

    result = {
        "generated_sql": corrected_sql,
        "correction_attempts": new_attempts,
    }

    if new_attempts >= max_corrections:
        result["pipeline_status"] = "failed"
        result["final_sql"] = corrected_sql

    return result