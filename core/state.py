"""
core/state.py — Unified state schema for all agents.
Every key is Optional so subgraphs only read what they need.
"""
from typing import Annotated, TypedDict, Sequence, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MasterState(TypedDict):
    # ── Conversation ─────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    user_feedback: Optional[str]

    # ── File tracking (per-user isolated paths) ──────────────────────
    file_paths: List[str]
    working_files: Dict[str, str]       # filename -> pickle path
    user_id: str                        # for per-user storage isolation

    # ── Pipeline control ─────────────────────────────────────────────
    error: Optional[str]
    iteration_count: int
    next_step: Optional[str]
    python_code: Optional[str]
    is_large_dataset: bool
    pending_human_question: Optional[str]

    # ── Active agent tracking (for frontend display) ─────────────────
    active_agent: Optional[str]         # which agent is currently running
    pipeline_status: Optional[str]      # "running" | "paused" | "complete" | "error"
    current_task_title: Optional[str]

    # ── Ingestion / Merge ────────────────────────────────────────────
    suggestion: Optional[str]

    # ── Preprocessing (clean agent) ──────────────────────────────────
    deep_profile_report: Optional[str]
    cleaning_plan: Optional[str]
    data_quality_score: Optional[float] # 0.0–1.0 score after cleaning

    # ── Task planner ─────────────────────────────────────────────────
    task_plan: Optional[str]            # JSON list of planned tasks
    current_task_index: int
    task_results: List[Dict[str, Any]]  # accumulated results per task
    dataset_issues: Optional[str]       # JSON summary of issues found during planning
    cleaning_strategy: Optional[str]    # JSON list of planned cleaning actions

    # ── ML agent ─────────────────────────────────────────────────────
    ml_model_path: Optional[str]        # path to saved model pickle
    ml_report: Optional[str]           # JSON report of metrics/predictions
    ml_task_type: Optional[str]        # "regression" | "classification" | "forecast"
    feature_importance: Optional[str]  # JSON

    # ── Chat / analysis ──────────────────────────────────────────────
    df_info: Optional[str]
    analysis_plan: Optional[str]
    analysis_result: Optional[str]     # last analysis / query result
    charts_generated: List[str]

    # ── Statistics & Insights ────────────────────────────────────────
    statistical_tests: Optional[str]   # JSON results of scipy stats tests
    insights: List[str]                # auto-generated business insights

    # ── Export ───────────────────────────────────────────────────────
    export_path: Optional[str]         # path to exported CSV/Excel

    # ── Report ───────────────────────────────────────────────────────
    report_path: Optional[str]         # path to generated HTML/PDF report

    # ── Resume capability ────────────────────────────────────────────
    user_checkpoint_id: Optional[str]
    last_checkpoint_at: Optional[str]
    agent_log: List[Dict[str, Any]]     # structured activity log for frontend
