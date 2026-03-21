"""
core/super_agent.py — Master Orchestrator (Maya)

Routes between all specialist agents based on intent and state.
Full pipeline:
  Upload → Ingest → Merge → Clean → Feature Engineer → ML → Chat

Also handles planner mode for complex multi-step goals.

FIXES:
- Added statistical_test, insights, export routes
- messages cleared when dispatching planner tasks to avoid context overflow
- Better keyword sets for routing accuracy
"""
import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from core.state import MasterState
from core.llm import get_llm
from core.activity_log import make_log_entry

logger = logging.getLogger(__name__)

router_llm = get_llm("fast", temperature=0.0)

# ── Keyword maps for fast routing ─────────────────────────────────────
_CLEAN_KW = {
    "clean", "fill", "impute", "drop", "deduplicate", "rename",
    "fix", "normalize", "standardize", "preprocess", "missing values",
    "outlier", "duplicate", "strip whitespace", "data quality",
}
_ML_KW = {
    "predict", "forecast", "model", "train", "classify", "regression",
    "accuracy", "features", "sales next", "inventory", "demand", "churn",
    "machine learning", "ml", "neural", "random forest", "xgboost",
    "lightgbm", "deep learning", "prophet", "arima", "time series model",
}
_FE_KW = {
    "feature", "engineer", "derive", "create column", "lag", "rolling",
    "ratio", "cyclical", "interaction", "encode", "new feature",
    "polynomial", "binning", "frequency encoding",
}
_CHAT_KW = {
    "show", "chart", "plot", "graph", "query", "summarize", "count",
    "average", "top", "distribution", "analyze", "compare", "how many",
    "what is", "list", "table", "correlation", "segment", "segmentation",
    "cluster", "clustering", "group by", "breakdown", "who are",
    "which", "describe", "explain", "tell me", "find", "identify",
    "t-test", "anova", "chi-square", "hypothesis", "p-value",
    "insight", "findings", "pattern", "anomaly",
}
_PLAN_KW = {
    "plan", "step by step", "build", "create a pipeline", "full analysis",
    "end to end", "complete analysis", "everything", "automate", "full pipeline",
    "do everything", "analyze everything", "full report", "comprehensive",
    "from scratch", "entire workflow",
}


def _already_merged(working_files: dict) -> bool:
    if not working_files:
        return False
    keys = list(working_files.keys())
    if any("merged" in k.lower() for k in keys):
        return True
    if len(keys) == 1:
        return True
    return False


def _quick_route(user_input: str, state: MasterState) -> str | None:
    txt           = user_input.lower()
    working_files = state.get("working_files", {})
    file_paths    = state.get("file_paths", [])

    # No files loaded yet → ingest first
    if file_paths and not working_files:
        return "ingestion"

    # Plan keywords always route to planner
    if any(kw in txt for kw in _PLAN_KW):
        return "planner"

    # Multiple unmerged files → merge first (unless already profiled/cleaned)
    if (working_files and len(working_files) > 1
            and not _already_merged(working_files)
            and not state.get("deep_profile_report")):
        return "merging"

    if any(kw in txt for kw in _ML_KW):
        return "ml"
    if any(kw in txt for kw in _FE_KW):
        return "feature_engineering"
    if any(kw in txt for kw in _CLEAN_KW):
        return "cleaning"
    if any(kw in txt for kw in _CHAT_KW):
        return "chat"

    return None


def entry_router(state: MasterState) -> str:
    user_input    = state.get("user_input", "")
    working_files = state.get("working_files", {})

    quick = _quick_route(user_input, state)
    if quick:
        logger.info(f"Quick route → {quick}")
        return quick

    # LLM routing for ambiguous cases
    context = (
        f'User request: "{user_input}"\n'
        f"State: files_loaded={bool(working_files)}, "
        f"data_cleaned={bool(state.get('deep_profile_report'))}, "
        f"model_trained={bool(state.get('ml_model_path'))}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Route to EXACTLY ONE of: ingestion|merging|cleaning|feature_engineering|ml|chat|planner

ingestion         — loading files, uploading data
merging           — joining multiple datasets
cleaning          — data quality, missing values, outliers
feature_engineering — creating new ML features
ml                — training models, predictions, forecasting
chat              — queries, stats, charts, questions, insights, statistical tests
planner           — complex multi-step goals, "do everything", full pipelines

Respond with EXACTLY ONE lowercase word."""),
        ("user", "{context}"),
    ])

    response = (prompt | router_llm).invoke({"context": context})
    intent   = response.content.lower().strip().strip("'\".")
    valid    = {"ingestion", "merging", "cleaning", "feature_engineering", "ml", "chat", "planner"}
    result   = intent if intent in valid else "chat"
    logger.info(f"LLM route → {result}")
    return result


def route_after_ingestion(state: MasterState) -> str:
    if state.get("error"):
        return END

    # If planner is running, return control to planner
    if (state.get("task_plan")
            and state.get("next_step")
            and state.get("next_step") != "done"):
        return "planner"

    user_input = state.get("user_input", "").lower()
    if any(kw in user_input for kw in _PLAN_KW):
        return "planner"

    working_files = state.get("working_files", {})
    if len(working_files) > 1:
        return "merging"
    return "cleaning"


def route_after_planner(state: MasterState) -> str:
    if state.get("error"):
        return END

    next_step = state.get("next_step", "")
    if not next_step or next_step in ("done", "finished"):
        return END

    mapping = {
        "ingest":           "ingestion",
        "merge":            "merging",
        "clean":            "cleaning",
        "feature_engineer": "feature_engineering",
        "ml_train":         "ml",
        "ml_predict":       "ml",
        "analyze":          "chat",
        "visualize":        "chat",
        # ★ New task types now routed correctly
        "statistical_test": "chat",
        "insights":         "chat",
        "export":           "chat",
        "report":           "chat",
    }
    return mapping.get(next_step, "chat")


def route_after_merging(state: MasterState) -> str:
    if (state.get("task_plan")
            and state.get("next_step")
            and state.get("next_step") not in ("done", "finished")):
        return "planner"
    return "cleaning"


def route_after_subagent(state: MasterState) -> str:
    """After any sub-agent: return to planner if it's running, else END."""
    if (state.get("task_plan")
            and state.get("next_step")
            and state.get("next_step") not in ("done", "finished")):
        return "planner"
    return END


def build_super_graph():
    from agents.ingestion.agent import build_ingestion_graph
    from agents.merging.merge_agent import build_merge_graph
    from agents.preprocessing.clean_agent import build_cleaning_graph
    from agents.feature_engineering.fe_agent import build_fe_graph
    from agents.ml.ml_agent import build_ml_graph
    from agents.chat.chat_agent import build_chat_graph
    from agents.planning.planner_agent import build_planner_graph

    workflow = StateGraph(MasterState)

    workflow.add_node("ingestion",           build_ingestion_graph().compile())
    workflow.add_node("merging",             build_merge_graph().compile())
    workflow.add_node("cleaning",            build_cleaning_graph().compile())
    workflow.add_node("feature_engineering", build_fe_graph().compile())
    workflow.add_node("ml",                  build_ml_graph().compile())
    workflow.add_node("chat",                build_chat_graph().compile())
    workflow.add_node("planner",             build_planner_graph().compile())

    workflow.add_conditional_edges(START, entry_router, {
        "ingestion":           "ingestion",
        "merging":             "merging",
        "cleaning":            "cleaning",
        "feature_engineering": "feature_engineering",
        "ml":                  "ml",
        "chat":                "chat",
        "planner":             "planner",
    })

    workflow.add_conditional_edges("ingestion", route_after_ingestion, {
        "merging":  "merging",
        "cleaning": "cleaning",
        "planner":  "planner",
        END: END,
    })

    workflow.add_conditional_edges("planner", route_after_planner, {
        "ingestion":           "ingestion",
        "merging":             "merging",
        "cleaning":            "cleaning",
        "feature_engineering": "feature_engineering",
        "ml":                  "ml",
        "chat":                "chat",
        "planner":             "planner",
        END: END,
    })

    workflow.add_conditional_edges("merging", route_after_merging, {
        "planner":  "planner",
        "cleaning": "cleaning",
        END: END,
    })

    for subagent in ["cleaning", "feature_engineering", "ml", "chat"]:
        workflow.add_conditional_edges(subagent, route_after_subagent, {
            "planner": "planner",
            END: END,
        })

    return workflow
