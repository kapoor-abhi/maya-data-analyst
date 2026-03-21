"""
agents/planning/planner_agent.py — Master Task Planner with HITL

Creates structured execution plans, shows them to the user for approval,
then dispatches task-by-task through the super_agent routing loop.

BUG FIXES:
- Replaced set_conditional_entry_point() (old API) with add_conditional_edges(START, ...)
- Removed var_child_runnable_config (internal API, not needed)
- Added statistical_test, insights, export task types
- messages cleared in get_next_task_node to prevent context overflow
"""
import os
import json
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.runnables.config import RunnableConfig

from core.state import MasterState
from core.llm import get_llm
from core.activity_log import make_log_entry

load_dotenv()
logger = logging.getLogger(__name__)

AUTO_APPROVE = os.getenv("AUTO_APPROVE", "").lower() in ("1", "true", "yes")
AUTO_APPROVE_PLAN = os.getenv("AUTO_APPROVE_PLAN", "").lower() in ("1", "true", "yes")

planner_llm = get_llm("coder", temperature=0.0)
fast_llm    = get_llm("fast",  temperature=0.0)

VALID_TASK_TYPES = {
    "ingest", "merge", "clean", "feature_engineer",
    "ml_train", "ml_predict", "analyze", "visualize",
    "statistical_test", "insights", "export", "report",
}


def _profile_datasets_for_planning(working_files: dict) -> tuple[list[dict], list[str]]:
    """Summarize likely data quality issues before the execution phase starts."""
    issues: list[dict] = []
    actions: list[str] = []

    for filename, path in working_files.items():
        try:
            df = pd.read_pickle(path)
        except Exception as exc:
            issues.append({
                "file": filename,
                "severity": "high",
                "title": "File could not be profiled",
                "detail": str(exc),
            })
            actions.append(f"Repair or reload `{filename}` before downstream tasks run.")
            continue

        duplicate_rows = int(df.duplicated().sum())
        missing_cols = []
        mixed_type_cols = []
        whitespace_cols = []
        datetime_like_cols = []
        constant_cols = []

        for col in df.columns:
            series = df[col]
            null_pct = round(float(series.isna().mean() * 100), 1)
            if null_pct > 0:
                missing_cols.append(f"{col} ({null_pct}% missing)")

            if series.notna().any() and series.dropna().nunique() == 1:
                constant_cols.append(col)

            if str(series.dtype) == "object":
                stripped = series.dropna().astype(str).str.strip()
                if len(stripped):
                    numeric_ratio = pd.to_numeric(stripped, errors="coerce").notna().mean()
                    if 0.1 < numeric_ratio < 0.9:
                        mixed_type_cols.append(col)
                    if (stripped != series.dropna().astype(str)).any():
                        whitespace_cols.append(col)
                if any(token in col.lower() for token in ["date", "time", "created", "updated", "timestamp"]):
                    datetime_like_cols.append(col)

        if duplicate_rows:
            issues.append({
                "file": filename,
                "severity": "medium",
                "title": "Duplicate rows detected",
                "detail": f"{duplicate_rows} duplicate rows need review.",
            })
            actions.append(f"Review exact duplicates in `{filename}` before analysis and modeling.")

        if missing_cols:
            issues.append({
                "file": filename,
                "severity": "high",
                "title": "Missing values found",
                "detail": ", ".join(missing_cols[:6]),
            })
            actions.append(f"Profile missingness in `{filename}` and use column-specific imputations.")

        if mixed_type_cols:
            issues.append({
                "file": filename,
                "severity": "medium",
                "title": "Mixed numeric/text values",
                "detail": ", ".join(mixed_type_cols[:6]),
            })
            actions.append(f"Normalize mixed-type columns in `{filename}` before feature engineering.")

        if whitespace_cols:
            issues.append({
                "file": filename,
                "severity": "low",
                "title": "Whitespace inconsistencies",
                "detail": ", ".join(whitespace_cols[:6]),
            })
            actions.append(f"Trim whitespace in `{filename}` to prevent duplicate category variants.")

        if constant_cols:
            issues.append({
                "file": filename,
                "severity": "low",
                "title": "Constant columns present",
                "detail": ", ".join(constant_cols[:6]),
            })
            actions.append(f"Review constant columns in `{filename}` and drop low-signal fields if needed.")

        if datetime_like_cols:
            issues.append({
                "file": filename,
                "severity": "medium",
                "title": "Datetime parsing review",
                "detail": ", ".join(datetime_like_cols[:6]),
            })
            actions.append(f"Parse datetime-like columns in `{filename}` before time-based analysis.")

    return issues, list(dict.fromkeys(actions))[:8]

PLANNER_SYSTEM = """You are Maya, a senior AI data scientist acting as a project planner.

Given the user's goal and current dataset state, create a precise ordered task plan.

OUTPUT FORMAT — respond ONLY with a valid JSON array, no other text:
[
  {
    "task_id": "t1",
    "task_type": "<ingest|merge|clean|feature_engineer|ml_train|ml_predict|analyze|visualize|statistical_test|insights|export|report>",
    "title": "Short title (max 6 words)",
    "description": "Exact instruction for the executing agent — be specific and actionable",
    "depends_on": [],
    "estimated_complexity": "low|medium|high",
    "estimated_duration": "~30s|~1min|~2min|~5min|~10min"
  }
]

PLANNING RULES:
1. Always start with ingest if files aren't loaded yet.
2. Always clean before ml_train, analyze, or statistical_test.
3. feature_engineer must come after clean and before ml_train.
4. Prediction/forecasting pipeline: clean → feature_engineer → ml_train → ml_predict
5. Charts should follow queries: analyze first, then visualize.
6. Keep tasks atomic — one clear action per task.
7. The description field is the EXACT instruction the agent receives. Be specific.
8. Maximum 12 tasks per plan.
9. If data is already cleaned, skip clean unless user explicitly asks.
10. Use statistical_test for hypothesis tests (t-test, ANOVA, chi-square).
11. Use insights to auto-generate key business findings.
12. Use export to save results to CSV/Excel.
"""


def _parse_plan(raw: str) -> list[dict]:
    """Extract JSON array from LLM response, tolerating markdown fences."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            plan = json.loads(match.group())
            for t in plan:
                if t.get("task_type") not in VALID_TASK_TYPES:
                    t["task_type"] = "analyze"
            return plan
        except json.JSONDecodeError:
            pass
    return []


def create_plan_node(state: MasterState) -> dict:
    """LLM creates the full task plan based on user goal + data state."""
    user_goal     = state.get("user_input", "")
    working_files = state.get("working_files", {})
    has_profile   = bool(state.get("deep_profile_report"))
    has_model     = bool(state.get("ml_model_path"))
    file_list     = list(working_files.keys())
    log_entries   = list(state.get("agent_log", []))
    dataset_issues, cleaning_strategy = _profile_datasets_for_planning(working_files) if working_files else ([], [])

    log_entries.append(make_log_entry(
        "Maya", "Creating execution plan", f"Goal: {user_goal[:100]}", "running",
    ))

    context = (
        f"User Goal: {user_goal}\n\n"
        f"Current State:\n"
        f"- Files loaded: {file_list if file_list else 'None — needs ingestion first'}\n"
        f"- Data cleaned/profiled: {has_profile}\n"
        f"- ML model trained: {has_model}\n"
        f"- File count: {len(file_list)}\n"
        f"- Planning issues found: {len(dataset_issues)}\n"
        f"- Suggested cleaning actions: {cleaning_strategy[:5]}\n"
    )

    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=context),
    ])
    plan = _parse_plan(response.content)

    if not plan:
        plan = [{
            "task_id": "t1",
            "task_type": "analyze",
            "title": "Answer user question",
            "description": user_goal,
            "depends_on": [],
            "estimated_complexity": "low",
            "estimated_duration": "~30s",
        }]

    logger.info(f"Planner created {len(plan)} tasks")
    log_entries.append(make_log_entry(
        "Maya", "Plan created", f"{len(plan)} tasks planned", "success",
        {"task_count": len(plan)},
    ))
    if dataset_issues:
        log_entries.append(make_log_entry(
            "Maya", "Dataset issues profiled",
            f"Found {len(dataset_issues)} issues before execution.", "success",
        ))

    return {
        "task_plan":           json.dumps(plan, indent=2),
        "current_task_index":  0,
        "task_results":        [],
        "current_task_title":  None,
        "error":               None,
        "pipeline_status":     "planning",
        "active_agent":        "planner",
        "agent_log":           log_entries,
        "dataset_issues":      json.dumps(dataset_issues, indent=2),
        "cleaning_strategy":   json.dumps(cleaning_strategy, indent=2),
    }


async def present_plan_node(state: MasterState, config: RunnableConfig) -> dict:
    """Show plan to user for approval — key HITL checkpoint."""
    plan = json.loads(state.get("task_plan", "[]"))
    dataset_issues = json.loads(state.get("dataset_issues", "[]") or "[]")
    cleaning_strategy = json.loads(state.get("cleaning_strategy", "[]") or "[]")

    icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    type_icons = {
        "ingest": "📥", "merge": "🔀", "clean": "🧹",
        "feature_engineer": "⚙️", "ml_train": "🤖", "ml_predict": "🔮",
        "analyze": "🔍", "visualize": "📊", "statistical_test": "📐",
        "insights": "💡", "export": "💾", "report": "📝",
    }

    plan_display = f"## 📋 Maya's Execution Plan\n\n"
    plan_display += f"I've created a **{len(plan)}-step plan**:\n\n"

    if dataset_issues:
        plan_display += "### Dataset issues found during planning\n"
        for issue in dataset_issues[:8]:
            plan_display += (
                f"- **{issue.get('file', 'dataset')}**: {issue.get('title', 'Issue')}"
                f" — {issue.get('detail', '')}\n"
            )
        plan_display += "\n"

    if cleaning_strategy:
        plan_display += "### How Maya plans to solve the data issues\n"
        for action in cleaning_strategy[:6]:
            plan_display += f"- {action}\n"
        plan_display += "\n"

    for i, task in enumerate(plan, 1):
        comp_icon  = icons.get(task.get("estimated_complexity", "medium"), "⚪")
        type_icon  = type_icons.get(task.get("task_type", "analyze"), "🔧")
        duration   = task.get("estimated_duration", "")
        plan_display += (
            f"**{i}. {type_icon} {task['title']}** `{task['task_type']}`"
            f" {comp_icon} {duration}\n"
            f"   _{task['description']}_\n\n"
        )

    plan_display += (
        "---\n"
        "Type **'approve'** to execute this plan, or describe any changes.\n"
        "You can say: 'change task 2 to...', 'skip task 3', or 'add a task to export results'."
    )

    if AUTO_APPROVE_PLAN:
        logger.info("AUTO_APPROVE: auto-approving plan")
        return {
            "user_feedback": "approve",
            "error": None,
            "pipeline_status": "running",
            "active_agent": "planner",
        }

    feedback = interrupt(plan_display)
    return {
        "user_feedback": feedback,
        "error": None,
        "pipeline_status": "waiting",
        "active_agent": "planner",
    }


def revise_plan_node(state: MasterState) -> dict:
    """Revise plan based on user feedback."""
    feedback      = state.get("user_feedback", "")
    original_plan = state.get("task_plan", "[]")
    user_goal     = state.get("user_input", "")

    response = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"Original goal: {user_goal}\n\n"
            f"Original plan:\n{original_plan}\n\n"
            f"User feedback / requested changes:\n{feedback}\n\n"
            f"Produce a revised plan incorporating the user's changes exactly."
        )),
    ])
    plan = _parse_plan(response.content)
    if not plan:
        plan = json.loads(original_plan)

    return {
        "task_plan": json.dumps(plan, indent=2),
        "current_task_index": 0,
        "current_task_title": None,
        "active_agent": "planner",
        "pipeline_status": "planning",
    }


def get_next_task_node(state: MasterState) -> dict:
    """Advance to the next task and set user_input for the sub-agent."""
    plan        = json.loads(state.get("task_plan", "[]"))
    idx         = state.get("current_task_index", 0)
    log_entries = list(state.get("agent_log", []))

    if idx >= len(plan):
        return {
            "next_step": "done",
            "error": None,
            "current_task_title": None,
            "active_agent": "planner",
        }

    task = plan[idx]
    logger.info(f"Task {idx + 1}/{len(plan)}: {task['title']}")
    log_entries.append(make_log_entry(
        "Maya", f"Starting: {task['title']}",
        task["description"][:100], "running",
        {"task_type": task["task_type"], "task_id": task["task_id"]},
    ))

    return {
        "user_input":      task["description"],
        "next_step":       task["task_type"],
        "error":           None,
        "iteration_count": 0,
        # ★ Clear messages to avoid context overflow when chaining many tasks
        "messages":        [],
        "agent_log":       log_entries,
        "active_agent":    "planner",
        "pipeline_status": "running",
        "current_task_title": task["title"],
    }


def record_task_result_node(state: MasterState) -> dict:
    """Record outcome of completed task and advance index."""
    plan        = json.loads(state.get("task_plan", "[]"))
    idx         = state.get("current_task_index", 0)
    results     = list(state.get("task_results", []))
    log_entries = list(state.get("agent_log", []))

    if idx < len(plan):
        task   = plan[idx]
        status = "error" if state.get("error") else "success"
        results.append({
            "task_id":   task["task_id"],
            "title":     task["title"],
            "task_type": task["task_type"],
            "status":    status,
            "error":     state.get("error"),
        })
        log_entries.append(make_log_entry(
            "Maya", f"Task complete: {task['title']}",
            state.get("error", ""), status,
        ))

    return {
        "task_results":       results,
        "current_task_index": idx + 1,
        "error":              None,
        "agent_log":          log_entries,
        "active_agent":       "planner",
        "pipeline_status":    "running",
    }


async def task_error_review_node(state: MasterState, config: RunnableConfig) -> dict:
    """Human review when a task keeps failing."""
    plan = json.loads(state.get("task_plan", "[]"))
    idx  = state.get("current_task_index", 0)
    task = plan[idx] if idx < len(plan) else {}

    msg = (
        f"⚠️ Task **{task.get('title', 'Unknown')}** failed after 3 attempts.\n\n"
        f"**Error:** `{state.get('error')}`\n\n"
        f"Options:\n"
        f"- Type a **corrected instruction** to retry with a new approach\n"
        f"- Type **'skip'** to skip this task and continue with the next\n"
        f"- Type **'abort'** to stop the pipeline"
    )

    if AUTO_APPROVE:
        logger.info(f"AUTO_APPROVE: skipping failed task '{task.get('title', '?')}'")
        return {
            "user_feedback": "skip",
            "error": None,
            "iteration_count": 0,
            "active_agent": "planner",
            "pipeline_status": "running",
        }

    feedback = interrupt(msg)
    return {
        "user_feedback": feedback,
        "error": None,
        "iteration_count": 0,
        "active_agent": "planner",
        "pipeline_status": "waiting",
    }


async def final_summary_node(state: MasterState, config: RunnableConfig) -> dict:
    """Generate and show final execution summary."""
    results   = state.get("task_results", [])
    ml_report = state.get("ml_report")

    success_count = sum(1 for r in results if r["status"] == "success")
    summary = f"## ✅ Pipeline Complete!\n\n**{success_count}/{len(results)} tasks succeeded**\n\n"
    summary += "### Execution Summary\n"

    for r in results:
        icon = "✅" if r["status"] == "success" else "❌"
        summary += f"{icon} **{r['title']}** (`{r['task_type']}`)"
        if r.get("error"):
            summary += f" — _{r['error'][:80]}_"
        summary += "\n"

    if ml_report:
        try:
            report = json.loads(ml_report)
            summary += "\n### 🤖 ML Results\n"
            for k, v in report.get("metrics", {}).items():
                summary += f"- **{k}:** `{v}`\n"
            if report.get("best_model"):
                summary += f"- **Best Model:** {report['best_model']}\n"
        except Exception:
            pass

    summary += "\n---\nYou can now ask questions about the data, request charts, or train new models."

    if AUTO_APPROVE:
        logger.info("AUTO_APPROVE: auto-completing final summary")
        return {"user_feedback": "done", "next_step": "finished",
                "pipeline_status": "complete", "active_agent": "planner"}

    feedback = interrupt(summary)
    return {"user_feedback": feedback, "next_step": "finished",
            "pipeline_status": "complete", "active_agent": "planner"}


# ── Routing ───────────────────────────────────────────────────────────

def planner_entry(state: MasterState) -> str:
    """Decide where to enter the planner subgraph."""
    if not state.get("task_plan"):
        return "create_plan"
    if state.get("next_step") == "done":
        return "final_summary"
    if state.get("error") and state.get("iteration_count", 0) >= 3:
        return "task_error_review"
    return "record_result"


def route_after_plan_review(state: MasterState) -> str:
    feedback = str(state.get("user_feedback", "")).strip().lower()
    if feedback in ["approve", "yes", "ok", "proceed", "run", "execute", "go", ""]:
        return "get_next_task"
    return "revise_plan"


def route_after_error_review(state: MasterState) -> str:
    feedback = str(state.get("user_feedback", "")).strip().lower()
    if feedback == "abort":
        return "final_summary"
    if feedback == "skip":
        return "record_result"
    return "get_next_task"


def route_task_dispatch(state: MasterState) -> str:
    """Compatibility helper used by tests and higher-level orchestration."""
    next_step = state.get("next_step", "")
    if next_step in ("done", "finished", ""):
        return "final_summary"

    mapping = {
        "ingest": "ingestion",
        "merge": "merging",
        "clean": "cleaning",
        "feature_engineer": "feature_engineering",
        "ml_train": "ml",
        "ml_predict": "ml",
        "analyze": "chat",
        "visualize": "chat",
        "statistical_test": "chat",
        "insights": "chat",
        "export": "chat",
        "report": "chat",
    }
    return mapping.get(next_step, "chat")


def build_planner_graph():
    workflow = StateGraph(MasterState)

    workflow.add_node("create_plan",       create_plan_node)
    workflow.add_node("present_plan",      present_plan_node)
    workflow.add_node("revise_plan",       revise_plan_node)
    workflow.add_node("get_next_task",     get_next_task_node)
    workflow.add_node("record_result",     record_task_result_node)
    workflow.add_node("task_error_review", task_error_review_node)
    workflow.add_node("final_summary",     final_summary_node)

    # ★ BUG FIX: set_conditional_entry_point() is old LangGraph API — removed.
    # Use add_conditional_edges(START, ...) instead.
    workflow.add_conditional_edges(START, planner_entry, {
        "create_plan":       "create_plan",
        "record_result":     "record_result",
        "task_error_review": "task_error_review",
        "final_summary":     "final_summary",
    })

    workflow.add_edge("create_plan", "present_plan")
    workflow.add_conditional_edges("present_plan", route_after_plan_review,
                                   {"get_next_task": "get_next_task",
                                    "revise_plan":   "revise_plan"})
    workflow.add_edge("revise_plan", "present_plan")
    workflow.add_edge("get_next_task", END)     # super_graph dispatches sub-agents
    workflow.add_edge("record_result", "get_next_task")
    workflow.add_conditional_edges("task_error_review", route_after_error_review,
                                   {"get_next_task":  "get_next_task",
                                    "record_result":  "record_result",
                                    "final_summary":  "final_summary"})
    workflow.add_edge("final_summary", END)

    return workflow
