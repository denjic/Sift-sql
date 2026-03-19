"""Phase 2: Chain-of-Thought Query Planning Agent node.

Emits an explicit natural-language logical blueprint that identifies all
structural elements required by the query, before any SQL syntax is generated.
"""

from text2sql_mas.llm.groq_client import GroqClient
from text2sql_mas.llm.prompts import PLANNING_PROMPT_V1
from text2sql_mas.schema.formatter import format_pruned_schema
from text2sql_mas.config import config


def query_planning_node(state: dict) -> dict:
    """
    LangGraph node: Phase 2 — CoT Blueprint Generation.

    Input state keys: nl_query, pruned_schema
    Output state keys: logical_blueprint
    """
    nl_query = state["nl_query"]
    pruned_schema = state["pruned_schema"]

    # Format the pruned schema for injection into the prompt
    pruned_schema_str = format_pruned_schema(
        list(pruned_schema.keys()), pruned_schema
    )

    # Build the planning prompt
    system_prompt = PLANNING_PROMPT_V1.format(
        pruned_schema=pruned_schema_str,
        nl_query=nl_query,
    )

    # Call Groq LLM for planning
    client = GroqClient(config)
    logical_blueprint = client.complete(
        system_prompt=system_prompt,
        user_message=nl_query,
    )

    return {
        "logical_blueprint": logical_blueprint,
    }
