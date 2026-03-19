"""Conditional edge routing functions for the LangGraph pipeline.

FIX LOG (v2):
- BUG FIXED: The original router never set final_sql = generated_sql on the
  "validated" path. When ast_validation_passed=True, pipeline_status and
  final_sql were never written, so run_pipeline() received final_sql=None.
  The metrics function then compared None against the gold SQL — always False —
  producing near-zero EX on the benchmark while Tab 1 appeared to work fine
  (Tab 1 reads generated_sql directly from the trace, not final_sql).
- ADDED: Explicit final_sql and pipeline_status assignment on all three paths
  so the state is always clean when the graph reaches END.
"""

from text2sql_mas.graph.state import SQLAgentState


def route_after_guard(state: SQLAgentState) -> str:
    """
    Conditional edge router: fires after the AST Guard node.

    Also writes final_sql and pipeline_status into state on terminal paths
    so the caller always receives a clean, populated state dict.

    Returns:
        "validated"  — AST passed; route to END
        "correct"    — AST failed, retries remaining; route to self_corrector
        "exhausted"  — AST failed, max retries reached; route to END (failed)
    """
    passed   = state.get("ast_validation_passed", False)
    attempts = state.get("correction_attempts", 0)
    max_corr = state.get("max_corrections", 3)

    if passed:
        # Terminal success — write final outputs into state before END
        state["final_sql"]       = state.get("generated_sql", "")
        state["pipeline_status"] = "success"
        return "validated"

    elif attempts < max_corr:
        # Still have retries left — go to self_corrector
        state["pipeline_status"] = "correction_loop"
        return "correct"

    else:
        # Max retries exhausted — write best-attempt and fail gracefully
        state["final_sql"]       = state.get("generated_sql", "")
        state["pipeline_status"] = "failed"
        return "exhausted"