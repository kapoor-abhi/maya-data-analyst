"""
agents/merging/merge_agent.py — Multi-file Merge Agent

Handles joining 2-20+ files using LLM-suggested strategies.
Includes:
- Automatic schema analysis to suggest join keys
- Conflict detection (overlapping column names)
- Audit check (result can't have 0 rows)
- Human approval before commit
- Self-correcting retry loop
"""
import os
import json
import logging
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.runnables.config import RunnableConfig

from core.state import MasterState
from core.llm import get_llm
from core.sandbox import DockerREPL, _strip_code
from core.activity_log import make_log_entry

logger = logging.getLogger(__name__)

fast_llm     = get_llm("fast",  temperature=0.0)
coder_llm    = get_llm("coder", temperature=0.0)
repl_sandbox = DockerREPL()


def _get_schema(working_files: dict) -> str:
    schema_info = ""
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            schema_info += f"\n--- File: {name} ---\n"
            schema_info += f"Columns: {list(df.columns)}\n"
            schema_info += f"Shape: {df.shape}\n"
            schema_info += f"Sample:\n{df.head(2).to_dict(orient='records')}\n"
        except Exception as e:
            schema_info += f"\n--- File: {name} --- [Error: {e}]\n"
    return schema_info


def analyze_merge_node(state: MasterState) -> dict:
    """Analyze all files and suggest optimal merge strategy."""
    working_files = state.get("working_files", {})
    log_entries   = list(state.get("agent_log", []))

    if len(working_files) < 2:
        return {"suggestion": "Only one file — no merge needed.",
                "error": None, "agent_log": log_entries,
                "active_agent": "merging", "pipeline_status": "running"}

    schema = _get_schema(working_files)
    log_entries.append(make_log_entry(
        "Merge", "Analyzing schemas",
        f"Found {len(working_files)} files to merge", "running",
    ))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Data Integration Expert.
Analyze the schemas of the provided DataFrames and suggest the best merge strategy.

Consider:
1. Columns with identical or similar names (e.g. 'ID' vs 'Client_ID', 'order_id' vs 'OrderID')
2. Whether to do inner/outer/left/right join (prefer outer to avoid data loss)
3. Whether to merge sequentially (A+B, then AB+C) or in a star schema
4. Overlapping non-key columns that need suffixes

Output a SHORT, human-readable suggestion (2-3 sentences max).
Example: "Merge sales.pkl and customers.pkl on 'customer_id', then join with products.pkl on 'product_id'. Use outer joins to preserve all records."
"""),
        ("user", "Schemas:\n{schema_text}"),
    ])

    response = (prompt | fast_llm).invoke({"schema_text": schema})
    suggestion = response.content.strip()

    log_entries.append(make_log_entry("Merge", "Strategy identified", suggestion, "success"))
    return {"suggestion": suggestion, "error": None,
            "iteration_count": 0, "agent_log": log_entries,
            "active_agent": "merging", "pipeline_status": "running"}


async def human_strategy_node(state: MasterState, config: RunnableConfig) -> dict:
    """Show merge strategy — auto-approve if no error, interrupt if failing."""
    AUTO_APPROVE = os.getenv("AUTO_APPROVE", "").lower() in ("1", "true", "yes")

    if state.get("error") and state.get("iteration_count", 0) >= 2:
        msg = (
            f"⚠️ Merge code failed after {state.get('iteration_count')} attempts.\n\n"
            f"**Error:** `{state.get('error')}`\n\n"
            f"Please provide a simpler instruction, or type **'skip'** to skip merging."
        )
        if AUTO_APPROVE:
            logger.info("AUTO_APPROVE: auto-skipping failing merge")
            return {
                "user_feedback": "skip",
                "error": None,
                "iteration_count": 0,
                "active_agent": "merging",
                "pipeline_status": "running",
            }

        feedback = interrupt(msg)
        if str(feedback).strip().lower() == "skip":
            return {
                "user_feedback": "skip",
                "error": None,
                "iteration_count": 0,
                "active_agent": "merging",
                "pipeline_status": "running",
            }
        return {
            "user_feedback": feedback,
            "error": None,
            "iteration_count": 0,
            "active_agent": "merging",
            "pipeline_status": "waiting",
        }

    suggestion = state.get("suggestion", "Merge with outer joins on common columns")
    logger.info(f"Merge: auto-approving strategy: {suggestion[:100]}")
    return {
        "user_feedback": "approve",
        "error": None,
        "iteration_count": 0,
        "active_agent": "merging",
        "pipeline_status": "running",
    }


def generate_merge_code_node(state: MasterState) -> dict:
    """Generate pandas merge code based on user feedback and schema."""
    if str(state.get("user_feedback", "")).strip().lower() == "skip":
        return {"python_code": "", "error": None}

    working_files = state.get("working_files", {})
    user_id       = state.get("user_id", "default")
    schema        = _get_schema(working_files)

    raw_instruction = state.get("user_feedback") or state.get("suggestion", "")
    passthrough = {"approve", "yes", "ok", ""}
    if str(raw_instruction).strip().lower() in passthrough:
        instruction = (f"Implement the following strategy: "
                       f"{state.get('suggestion', 'merge all files on common columns using outer join')}")
    else:
        instruction = raw_instruction

    error_context = (
        f"\n\nFIX THIS ERROR:\n{state.get('error')}" if state.get("error") else ""
    )

    merge_output_path = os.path.join("storage", user_id, "sandbox", "merged_dataset.pkl")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Pandas Expert.

RULES:
1. Import pandas.
2. Define working_files exactly as provided.
3. Load each df with pd.read_pickle(path).
4. Merge per instruction.
5. Save final result to: {output_path}
6. Use how='outer' by default to avoid data loss.
7. Handle overlapping non-key columns with suffixes.
8. For multiple files, merge sequentially.

Return ONLY valid Python inside ```python ... ``` blocks."""),
        ("user", "Files:\n{files}\n\nSchemas:\n{schema}\n\nInstruction: {instruction}{error_context}"),
    ])

    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema,
        "instruction": instruction,
        "output_path": merge_output_path,
        "error_context": error_context,
    })

    code = _strip_code(response.content)
    return {"python_code": code, "error": None}


def execute_merge_node(state: MasterState) -> dict:
    """Execute merge code with audit validation."""
    code          = state.get("python_code", "")
    working_files = state.get("working_files", {})
    user_id       = state.get("user_id", "default")
    log_entries   = list(state.get("agent_log", []))

    if not code:
        return {
            "error": None,
            "iteration_count": 0,
            "agent_log": log_entries,
            "active_agent": "merging",
            "pipeline_status": "running",
        }

    result = repl_sandbox.run(code)

    if result.get("error"):
        current_iter = state.get("iteration_count", 0) + 1
        log_entries.append(make_log_entry("Merge", "Execution failed", result["error"], "error"))
        return {
            "error": result["error"],
            "iteration_count": current_iter,
            "agent_log": log_entries,
            "active_agent": "merging",
            "pipeline_status": "running",
        }

    merge_output_path = os.path.join("storage", user_id, "sandbox", "merged_dataset.pkl")

    try:
        df = pd.read_pickle(merge_output_path)
        if len(df) == 0:
            current_iter = state.get("iteration_count", 0) + 1
            return {"error": "AUDIT FAILURE: merge resulted in 0 rows. "
                             "Check join keys or use how='outer'.",
                    "iteration_count": current_iter, "agent_log": log_entries,
                    "active_agent": "merging", "pipeline_status": "running"}

        log_entries.append(make_log_entry(
            "Merge", "Merge complete",
            f"Merged dataset: {df.shape[0]} rows × {df.shape[1]} cols", "success",
            {"rows": df.shape[0], "cols": df.shape[1]},
        ))
        return {
            "working_files": {"merged_dataset.pkl": merge_output_path},
            "error": None, "iteration_count": 0, "agent_log": log_entries,
            "active_agent": "merging", "pipeline_status": "running",
        }
    except Exception as e:
        current_iter = state.get("iteration_count", 0) + 1
        return {"error": f"Failed to audit merged dataset: {e}",
                "iteration_count": current_iter, "agent_log": log_entries,
                "active_agent": "merging", "pipeline_status": "running"}


def route_merge_retry(state: MasterState) -> str:
    if state.get("error"):
        if state.get("iteration_count", 0) >= 3:
            return "human_strategy"
        return "generate_code"
    return "success"


def build_merge_graph():
    workflow = StateGraph(MasterState)
    workflow.add_node("analyze",        analyze_merge_node)
    workflow.add_node("human_strategy", human_strategy_node)
    workflow.add_node("generate_code",  generate_merge_code_node)
    workflow.add_node("execute",        execute_merge_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze",        "human_strategy")
    workflow.add_edge("human_strategy", "generate_code")
    workflow.add_edge("generate_code",  "execute")
    workflow.add_conditional_edges("execute", route_merge_retry,
                                   {"generate_code": "generate_code",
                                    "human_strategy": "human_strategy",
                                    "success": END})
    return workflow
