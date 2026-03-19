"""Phase 3: Constrained SQL Generation Agent node.

Translates the logical blueprint and pruned schema into executable SQL
under deterministic generation constraints (temperature=0.0).
"""

from text2sql_mas.llm.groq_client import GroqClient
from text2sql_mas.llm.prompts import GENERATION_PROMPT_V1
from text2sql_mas.schema.formatter import format_pruned_schema
from text2sql_mas.config import config


def sql_generation_node(state: dict) -> dict:
    """
    LangGraph node: Phase 3 — Constrained SQL Generation.

    Input state keys: nl_query, pruned_schema, logical_blueprint
    Output state keys: generated_sql
    """
    nl_query = state["nl_query"]
    pruned_schema = state["pruned_schema"]
    logical_blueprint = state["logical_blueprint"]

    # Format pruned schema for the prompt
    pruned_schema_str = format_pruned_schema(
        list(pruned_schema.keys()), pruned_schema
    )

    # Build the generation prompt
    system_prompt = GENERATION_PROMPT_V1.format(
        pruned_schema=pruned_schema_str,
        logical_blueprint=logical_blueprint,
        nl_query=nl_query,
    )

    # Call Groq LLM for SQL generation
    client = GroqClient(config)
    generated_sql = client.complete(
        system_prompt=system_prompt,
        user_message=nl_query,
    )

    # Clean potential markdown formatting from LLM output
    generated_sql = generated_sql.strip()
    if generated_sql.startswith("```"):
        lines = generated_sql.split("\n")
        # Remove first and last lines if they are code fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        generated_sql = "\n".join(lines).strip()

    return {
        "generated_sql": generated_sql,
    }
