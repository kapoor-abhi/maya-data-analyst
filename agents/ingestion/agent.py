"""
agents/ingestion/agent.py — Data Ingestion Agent

Handles loading of CSV, Excel, Parquet, JSON files into pickle format.
Supports 10+ files simultaneously with per-file optimization.
Includes human review checkpoint after ingestion.
"""
import os
import json
import shutil
import logging
import pandas as pd
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.runnables.config import RunnableConfig

from core.state import MasterState
from core.llm import get_llm
from core.sandbox import DockerREPL, _strip_code
from core.activity_log import make_log_entry

logger = logging.getLogger(__name__)

coder_llm    = get_llm("coder", temperature=0.0)
repl_sandbox = DockerREPL()


def _get_schema(working_files: dict) -> str:
    schema_info = ""
    for name, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            schema_info += f"\n--- File: {name} ---\n"
            schema_info += f"Columns: {list(df.columns)}\n"
            schema_info += "Data Types:\n"
            for col, dtype in df.dtypes.items():
                sample = df[col].dropna().head(2).tolist()
                schema_info += f"  {col} ({dtype}): sample={sample}\n"
            schema_info += f"Shape: {df.shape}\n"
        except Exception as e:
            schema_info += f"\n--- File: {name} --- [Error: {e}]\n"
    return schema_info


def ingest_data_node(state: MasterState) -> dict:
    """Load all uploaded files into pickle format. Handles CSV, Excel, Parquet, JSON."""
    file_paths   = state.get("file_paths", [])
    working_files = dict(state.get("working_files", {}))
    user_id      = state.get("user_id", "default")

    log_entries = list(state.get("agent_log", []))
    log_entries.append(make_log_entry(
        "Ingestion", "Loading files",
        f"Processing {len(file_paths)} file(s)", "running",
    ))

    errors, loaded = [], []

    for path in file_paths:
        filename  = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]

        sandbox_dir = os.path.join("storage", user_id, "sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)
        pickle_path = os.path.join(sandbox_dir, f"{base_name}.pkl")

        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".csv":
                df = _smart_read_csv(path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
            elif ext == ".json":
                try:
                    df = pd.read_json(path)
                except Exception:
                    import json as _json
                    with open(path) as fh:
                        data = _json.load(fh)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
            else:
                df = _smart_read_csv(path)

            df.to_pickle(pickle_path)
            working_files[f"{base_name}.pkl"] = pickle_path
            loaded.append(filename)
            logger.info(f"Ingested {filename}: {df.shape}")

        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            logger.error(f"Failed to ingest {filename}: {e}", exc_info=True)

    if errors:
        log_entries.append(make_log_entry(
            "Ingestion", "Load complete with errors",
            f"Loaded: {loaded}, Errors: {errors}", "error",
        ))
        return {"working_files": working_files,
                "error": f"Some files failed: {'; '.join(errors)}",
                "agent_log": log_entries,
                "active_agent": "ingestion",
                "pipeline_status": "running"}

    log_entries.append(make_log_entry(
        "Ingestion", "All files loaded",
        f"Loaded {len(loaded)} files: {loaded}", "success",
        {"files": loaded, "count": len(loaded)},
    ))
    return {
        "working_files": working_files,
        "error": None,
        "agent_log": log_entries,
        "active_agent": "ingestion",
        "pipeline_status": "running",
    }


def _smart_read_csv(path: str) -> pd.DataFrame:
    """Smart CSV reader: detects encoding and separator."""
    for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                sample = f.read(4096)
            sep = ","
            for candidate in [",", ";", "\t", "|"]:
                if sample.count(candidate) > sample.count(sep):
                    sep = candidate
            df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
            if len(df.columns) > 1 or len(df) > 0:
                return df
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def optimize_data_node(state: MasterState) -> dict:
    """Optimize dataframe memory for large datasets."""
    working_files = state.get("working_files", {})
    is_large      = False
    optimized     = []

    for name, path in working_files.items():
        try:
            df     = pd.read_pickle(path)
            n_rows = len(df)

            if n_rows > 50_000:
                is_large  = True
                before_mb = df.memory_usage(deep=True).sum() / 1e6

                for col in df.select_dtypes(include=["object"]).columns:
                    if df[col].nunique() / n_rows < 0.5:
                        df[col] = df[col].astype("category")
                for col in df.select_dtypes(include=["int64"]).columns:
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                for col in df.select_dtypes(include=["float64"]).columns:
                    df[col] = pd.to_numeric(df[col], downcast="float")

                after_mb = df.memory_usage(deep=True).sum() / 1e6
                df.to_pickle(path)
                optimized.append(f"{name}: {before_mb:.1f}MB → {after_mb:.1f}MB")

        except Exception as e:
            logger.error(f"Optimize error for {name}: {e}")

    log_entries = list(state.get("agent_log", []))
    if optimized:
        log_entries.append(make_log_entry(
            "Ingestion", "Memory optimized", "; ".join(optimized), "success",
        ))

    return {
        "is_large_dataset": is_large,
        "agent_log": log_entries,
        "active_agent": "ingestion",
        "pipeline_status": "running",
    }


def column_selection_node(state: MasterState) -> dict:
    """LLM-driven column selection/transformation based on user instruction."""
    working_files = state.get("working_files", {})
    schema        = _get_schema(working_files)
    is_large      = state.get("is_large_dataset", False)

    raw_intent = state.get("user_feedback") or state.get("user_input", "")

    passthrough_cmds = {
        "approve", "yes", "ok", "", "load", "proceed", "continue",
        "load the data", "load and clean this data", "done",
    }
    if str(raw_intent).strip().lower().rstrip(".") in passthrough_cmds:
        return {"python_code": None, "error": None}

    performance_note = (
        "PERFORMANCE MODE: Use vectorized Pandas ops only, NO loops." if is_large else ""
    )
    error_context = (
        f"\n\nFIX THIS ERROR FROM PREVIOUS ATTEMPT:\n{state.get('error')}"
        if state.get("error") else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python Data Expert. Write code to transform the data per user instruction.

RULES:
1. Import pandas and numpy.
2. Define working_files dict exactly as provided.
3. Load with pd.read_pickle(path), transform, save with df.to_pickle(path) to SAME path.
4. NEVER drop rows (no dropna, no drop_duplicates) — that's for the cleaning phase.
5. NEVER cast datatypes — leave for cleaning phase.
6. NEVER create dummy data.
7. NO print() statements.
{performance_note}

Return ONLY valid Python inside ```python ... ``` blocks."""),
        ("user", "Files:\n{files}\n\nSchemas:\n{schema}\n\nInstruction: {query}{error_context}"),
    ])

    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema,
        "query": raw_intent,
        "performance_note": performance_note,
        "error_context": error_context,
    })

    code = _strip_code(response.content)
    return {"python_code": code, "error": None}


def execute_code_node(state: MasterState) -> dict:
    """Execute generated transformation code with rollback on failure."""
    code          = state.get("python_code")
    working_files = state.get("working_files", {})

    if not code:
        return {"error": None, "iteration_count": 0}

    # Backup files before execution
    for name, path in working_files.items():
        if os.path.exists(path):
            shutil.copy(path, path + ".bak")

    result = repl_sandbox.run(code)

    if result.get("error"):
        for name, path in working_files.items():
            bak = path + ".bak"
            if os.path.exists(bak):
                shutil.move(bak, path)
        return {
            "error": result["error"],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # Remove backups on success
    for name, path in working_files.items():
        bak = path + ".bak"
        if os.path.exists(bak):
            os.remove(bak)

    return {"error": None, "iteration_count": 0}


async def human_review_node(state: MasterState, config: RunnableConfig) -> dict:
    """Human review checkpoint after ingestion.
    
    Auto-approves for simple 'load data' instructions.
    Only interrupts when user gave meaningful transformation instructions.
    """
    AUTO_APPROVE = os.getenv("AUTO_APPROVE", "").lower() in ("1", "true", "yes")
    working_files = state.get("working_files", {})

    # If there was an error and AUTO_APPROVE, still auto-approve to let pipeline continue
    if AUTO_APPROVE:
        logger.info("AUTO_APPROVE: auto-approving ingestion review")
        return {"user_feedback": "approve", "error": None}

    schema = _get_schema(working_files)

    if state.get("error"):
        msg = (
            f"⚠️ **Some issues occurred:**\n{state['error']}\n\n"
            f"Files loaded: {list(working_files.keys())}\n\n"
            f"Type **'approve'** to continue anyway, or describe what to fix."
        )
    else:
        msg = (
            f"✅ **Files loaded successfully!**\n\n"
            f"**Files ready:** {list(working_files.keys())}\n\n"
            f"**Schema Preview:**\n```\n{schema[:1500]}\n```\n\n"
            f"Type **'approve'** to continue to data cleaning, "
            f"or describe any changes you need."
        )

    feedback = interrupt(msg)
    return {"user_feedback": feedback, "error": None}

# ── Routing ───────────────────────────────────────────────────────────

def route_after_ingestion(state: MasterState) -> str:
    if state.get("error"):
        return END
    return "optimize_data"


def route_execution(state: MasterState) -> str:
    if state.get("error"):
        if state.get("iteration_count", 0) >= 3:
            return "human_review"
        return "column_selection"
    return "human_review"


def route_after_review(state: MasterState) -> str:
    feedback = str(state.get("user_feedback", "")).strip().lower().rstrip(".")
    passthrough = {"yes", "y", "ok", "approve", "proceed", "looks good",
                   "continue", "done", ""}
    if feedback in passthrough:
        return "end"
    return "column_selection"


def build_ingestion_graph():
    workflow = StateGraph(MasterState)
    workflow.add_node("ingest_data",      ingest_data_node)
    workflow.add_node("optimize_data",    optimize_data_node)
    workflow.add_node("column_selection", column_selection_node)
    workflow.add_node("execute_code",     execute_code_node)
    workflow.add_node("human_review",     human_review_node)

    workflow.set_entry_point("ingest_data")
    workflow.add_conditional_edges("ingest_data", route_after_ingestion,
                                   {"optimize_data": "optimize_data", END: END})
    workflow.add_edge("optimize_data", "column_selection")
    workflow.add_edge("column_selection", "execute_code")
    workflow.add_conditional_edges("execute_code", route_execution,
                                   {"column_selection": "column_selection",
                                    "human_review":     "human_review"})
    workflow.add_conditional_edges("human_review", route_after_review,
                                   {"column_selection": "column_selection", "end": END})
    return workflow
