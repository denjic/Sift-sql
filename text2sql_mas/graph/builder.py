"""LangGraph StateGraph construction — wires up all agent nodes and conditional edges."""

from langgraph.graph import StateGraph, END

from text2sql_mas.graph.state import SQLAgentState
from text2sql_mas.agents.schema_linker import schema_linking_node
from text2sql_mas.agents.query_planner import query_planning_node
from text2sql_mas.agents.sql_generator import sql_generation_node
from text2sql_mas.agents.ast_guard import ast_guard_node
from text2sql_mas.agents.self_corrector import self_correction_node
from text2sql_mas.graph.router import route_after_guard


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
