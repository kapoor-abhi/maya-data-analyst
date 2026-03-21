"""
feature_engineering/fe_agent.py — Intelligent Feature Engineering Agent

Uses a ReAct loop to:
1. Inspect the dataset and understand the ML goal
2. Generate domain-appropriate features
3. Validate they improve signal (variance, correlation with target)
4. Self-correct if code errors occur

Handles: date decomposition, lag features, rolling statistics,
ratio features, polynomial interactions, cyclical encoding,
target encoding, binning, text features, domain-specific features.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.runnables.config import RunnableConfig

from core.state import MasterState
from core.llm import get_llm
from core.sandbox import DockerREPL, _strip_code
from core.activity_log import make_log_entry

logger = logging.getLogger(__name__)

coder_llm    = get_llm("coder", temperature=0.0)
MAX_FE_TURNS = 15
MAX_RESULT_CHARS = 4000


def _trunc(s: str, n: int = MAX_RESULT_CHARS) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s)-n} chars]"


def _load(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


# ── Tools available to the FE agent ──────────────────────────────────

def tool_inspect_for_fe(working_files: dict) -> dict:
    """Inspect dataset to identify FE opportunities."""
    result = {}
    for name, path in working_files.items():
        df = _load(path)
        numeric_stats = {}
        for col in df.select_dtypes(include=np.number).columns[:20]:
            s = df[col].dropna()
            numeric_stats[col] = {
                "mean": round(float(s.mean()), 4) if len(s) else None,
                "std":  round(float(s.std()),  4) if len(s) else None,
                "min":  round(float(s.min()),  4) if len(s) else None,
                "max":  round(float(s.max()),  4) if len(s) else None,
            }
        result[name] = {
            "shape":           list(df.shape),
            "columns":         {c: str(t) for c, t in df.dtypes.items()},
            "null_pcts":       {c: round(df[c].isnull().mean() * 100, 1) for c in df.columns},
            "datetime_cols":   df.select_dtypes(include="datetime64").columns.tolist(),
            "numeric_cols":    df.select_dtypes(include=np.number).columns.tolist(),
            "categorical_cols":df.select_dtypes(include="object").columns.tolist(),
            "numeric_stats":   numeric_stats,
            "sample":          df.head(3).to_dict(orient="records"),
        }
    return result


def tool_list_current_columns(working_files: dict) -> dict:
    """List all current columns (including newly created ones) in each file."""
    result = {}
    for name, path in working_files.items():
        df = _load(path)
        result[name] = {
            "columns": list(df.columns),
            "shape":   list(df.shape),
            "dtypes":  {c: str(t) for c, t in df.dtypes.items()},
        }
    return result


def tool_run_fe_code(working_files: dict, code: str, description: str) -> dict:
    """Execute feature engineering code in sandbox."""
    sandbox = DockerREPL()
    code    = _strip_code(code)

    import ast as _ast
    try:
        _ast.parse(code)
    except SyntaxError as e:
        return {"status": "error", "error": f"Syntax error: {e}"}

    preamble = (
        f"import pandas as pd\nimport numpy as np\n"
        f"working_files = {json.dumps(working_files)}\n"
    )
    result = sandbox.run(preamble + code)
    if result.get("error"):
        return {"status": "error", "error": result["error"]}
    return {"status": "success", "stdout": result.get("output", "")[:1000],
            "description": description}


def tool_validate_features(working_files: dict, filename: str,
                            new_cols: list, target_col: str = None) -> dict:
    """Check that new features have variance and optionally correlate with target."""
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    report = {}
    for col in new_cols:
        if col not in df.columns:
            report[col] = {"status": "MISSING — not created in the dataframe"}
            continue
        s = df[col].dropna()
        info = {
            "dtype":        str(df[col].dtype),
            "null_pct":     round(df[col].isnull().mean() * 100, 2),
            "nunique":      int(s.nunique()),
            "has_variance": (bool(s.std() > 0) if pd.api.types.is_numeric_dtype(s)
                             else bool(s.nunique() > 1)),
        }
        if (target_col and target_col in df.columns
                and pd.api.types.is_numeric_dtype(df[col])):
            try:
                corr = float(df[[col, target_col]].dropna().corr().iloc[0, 1])
                info["correlation_with_target"] = round(abs(corr), 4)
            except Exception:
                pass
        report[col] = info
    return report


FE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_for_fe",
            "description": "Inspect the dataset structure, data types, null rates, and stats to find FE opportunities. Call this FIRST.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_current_columns",
            "description": "List all current columns in each file including newly created ones. Use after run_fe_code to verify columns were created.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_fe_code",
            "description": (
                "Execute Python code to create new features. "
                "Code must load with pd.read_pickle(working_files[name]) and save back with df.to_pickle(path)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code":        {"type": "string", "description": "Python code to run"},
                    "description": {"type": "string", "description": "Summary of features created"},
                },
                "required": ["code", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_features",
            "description": "Validate newly created columns: check they exist, have variance, and optionally correlate with target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename":   {"type": "string", "description": "File to check"},
                    "new_cols":   {"type": "array", "items": {"type": "string"},
                                  "description": "List of new column names"},
                    "target_col": {"type": "string", "description": "Target column for correlation check"},
                },
                "required": ["filename", "new_cols"],
            },
        },
    },
]

FE_SYSTEM = """You are a senior feature engineer for machine learning.

Given a dataset and a goal (e.g. predict sales, classify churn), create the most useful features.

FEATURE TYPES TO CONSIDER:
1. Datetime: year, month, day_of_week, is_weekend, days_since_epoch, quarter, hour, day_of_year
2. Cyclical: sin/cos encoding for month (1-12), hour (0-23), day_of_week (0-6)
3. Lag features: previous period values for time-series (lag_1, lag_7, lag_30) — group by entity
4. Rolling statistics: rolling_mean_7, rolling_std_7, rolling_max_30 (group by entity, sort by time)
5. Ratio features: revenue/cost, clicks/impressions, profit_margin = (revenue-cost)/revenue
6. Interaction: multiply or add two semantically related columns
7. Binning: pd.cut or pd.qcut for continuous → categorical buckets (age groups, price tiers)
8. Text: char_length, word_count, contains_keyword, sentiment score for string columns
9. Frequency encoding: replace category with its count/frequency in the dataset
10. Polynomial: x^2 for key numeric predictors (use sparingly, only when clearly nonlinear)
11. Domain-specific: days_until_expiry, recency-frequency-monetary for retail
12. Aggregation: customer-level mean/max/count of order values (group by customer_id)

RULES:
- ALWAYS start with inspect_for_fe() to understand the data
- Use list_current_columns() after run_fe_code() to verify columns were created
- Code must load from pd.read_pickle(path) and save back with df.to_pickle(path)
- Validate every batch of features with validate_features() before proceeding
- Drop features with zero variance (add noise with no signal)
- Do NOT apply target encoding to the full dataset — only flag it as a recommendation
- Handle NaN carefully in lag/rolling features (use fillna or min_periods)
- When fully done with all features, say "FEATURE ENGINEERING COMPLETE" and list all new columns
"""


def dispatch_fe_tool(name: str, args: dict, working_files: dict) -> str:
    if name == "inspect_for_fe":
        result = tool_inspect_for_fe(working_files)
    elif name == "list_current_columns":
        result = tool_list_current_columns(working_files)
    elif name == "run_fe_code":
        result = tool_run_fe_code(
            working_files,
            args.get("code", ""),
            args.get("description", ""),
        )
    elif name == "validate_features":
        result = tool_validate_features(
            working_files,
            args.get("filename", next(iter(working_files), "")),
            args.get("new_cols", []),
            args.get("target_col"),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    try:
        s = json.dumps(result, default=str, indent=2)
        return _trunc(s)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agent nodes ───────────────────────────────────────────────────────

def fe_agent_node(state: MasterState) -> dict:
    working_files    = state.get("working_files", {})
    user_instruction = state.get("user_input", "Engineer features for ML prediction")
    messages         = list(state.get("messages", []))
    log_entries      = list(state.get("agent_log", []))

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [
            SystemMessage(content=FE_SYSTEM),
            HumanMessage(content=(
                f"Goal: {user_instruction}\n"
                f"Files: {list(working_files.keys())}\n"
                f"Start with inspect_for_fe() to understand the data."
            )),
        ]

    llm_with_tools = coder_llm.bind_tools(FE_TOOL_SCHEMAS)
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Log tool calls for frontend activity feed
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            log_entries.append(make_log_entry(
                "FeatureEngineer", f"Calling {tc['name']}",
                str(tc.get("args", {}))[:200], "running",
            ))

    return {
        "messages":       messages,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "error":          None,
        "agent_log":      log_entries,
        "active_agent":   "feature_engineering",
    }


def fe_tool_executor_node(state: MasterState) -> dict:
    messages      = list(state.get("messages", []))
    working_files = state.get("working_files", {})
    log_entries   = list(state.get("agent_log", []))
    last          = messages[-1]

    for tc in last.tool_calls:
        result = dispatch_fe_tool(tc["name"], tc["args"], working_files)
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        try:
            r = json.loads(result)
            status = "error" if r.get("status") == "error" or r.get("error") else "success"
            log_entries.append(make_log_entry(
                "FeatureEngineer", f"Tool result: {tc['name']}",
                str(r)[:200], status,
            ))
        except Exception:
            pass

    return {"messages": messages, "agent_log": log_entries}


async def fe_review_node(state: MasterState, config: RunnableConfig) -> dict:
    """Human review checkpoint after feature engineering."""
    messages     = state.get("messages", [])
    last_content = messages[-1].content if messages else ""
    msg = (
        f"⚙️ **Feature Engineering Complete!**\n\n"
        f"{last_content}\n\n"
        f"Type **'approve'** to continue to ML training, "
        f"or request additional features (e.g. 'also add rolling 14-day average of sales')."
    )
    feedback = interrupt(msg)
    return {"user_feedback": feedback, "error": None, "iteration_count": 0}


# ── Routing ───────────────────────────────────────────────────────────

def route_fe_agent(state: MasterState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return END

    last    = messages[-1]
    content = getattr(last, "content", "") or ""

    # Hard stop → go to review
    if state.get("iteration_count", 0) >= MAX_FE_TURNS:
        return "fe_review"

    # Has tool calls → execute them
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "fe_tools"

    # Agent declared done → go to review
    if "FEATURE ENGINEERING COMPLETE" in content.upper():
        return "fe_review"

    # ★ BUG FIX: default was "fe_review" which broke the ReAct loop entirely.
    # If no tool calls and not done, continue the loop so the agent can keep working.
    return "fe_agent"


def route_after_fe_review(state: MasterState) -> str:
    feedback = str(state.get("user_feedback", "")).strip().lower()
    if feedback in ["approve", "yes", "ok", "done", ""]:
        return END
    return "fe_agent"


def build_fe_graph():
    workflow = StateGraph(MasterState)
    workflow.add_node("fe_agent",  fe_agent_node)
    workflow.add_node("fe_tools",  fe_tool_executor_node)
    workflow.add_node("fe_review", fe_review_node)

    workflow.set_entry_point("fe_agent")
    workflow.add_conditional_edges("fe_agent", route_fe_agent,
                                   {"fe_tools": "fe_tools",
                                    "fe_review": "fe_review",
                                    "fe_agent":  "fe_agent",
                                    END: END})
    workflow.add_edge("fe_tools", "fe_agent")
    workflow.add_conditional_edges("fe_review", route_after_fe_review,
                                   {"fe_agent": "fe_agent", END: END})
    return workflow
