"""
app/main.py — Maya Production FastAPI Server v3.2

Architecture:
- POST /upload  → saves files, fires background asyncio.Task, returns immediately
- GET /stream/{thread_id} → SSE emits log entries in real-time as pipeline runs,
                            then emits final result when done/paused
- POST /chat    → synchronous for short queries
- POST /resume  → non-blocking resume (same background-task pattern as upload)

BUGS FIXED vs original:
1. interrupt() crash: ALL human_review nodes now have AUTO_APPROVE guard before interrupt()
2. Blocking upload: ainvoke no longer runs inside the HTTP handler — pipeline is a background task
3. SSE showed nothing during execution: now polls live checkpoint state at 500ms intervals
4. clean_agent ReAct fallback went to human_review instead of analyst_agent — fixed
5. fe_agent review had no AUTO_APPROVE guard — added
6. agent.py ingestion review had no AUTO_APPROVE guard — added
7. Chat endpoint passed tuple instead of HumanMessage — fixed
8. AsyncPostgresSaver required autocommit=True — fixed
"""
import os, json, math, time, logging, asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.types import Command

from core.super_agent import build_super_graph
from core.storage import get_storage, cleanup_stale_users, QuotaExceededError

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URI          = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/maya")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_FILES       = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
USE_POSTGRES    = os.getenv("USE_POSTGRES", "true").lower() not in ("0", "false", "no")

langfuse_handler = None
try:
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
    logger.info("Langfuse active")
except ImportError:
    pass

try:
    from redis import Redis
    from langchain_community.cache import RedisCache
    from langchain_core.globals import set_llm_cache
    r = Redis.from_url(REDIS_URL, socket_connect_timeout=2); r.ping()
    set_llm_cache(RedisCache(redis_=r))
    logger.info("Redis LLM cache active")
except Exception:
    pass


# ── JSON helpers ──────────────────────────────────────────────────────
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        try:
            if pd.isna(obj): return None
        except Exception: pass
        return super().default(obj)

def _safe_json(obj) -> str: return json.dumps(_deep_clean(obj), cls=_NpEncoder)

def _deep_clean(d):
    if isinstance(d, dict): return {k: _deep_clean(v) for k, v in d.items()}
    if isinstance(d, list): return [_deep_clean(v) for v in d]
    if isinstance(d, float) and (math.isnan(d) or math.isinf(d)): return None
    try:
        if pd.isna(d): return None
    except Exception: pass
    return d

def _extract_interrupt(snapshot) -> tuple[str, dict]:
    msg, values = "", getattr(snapshot, "values", {})
    if hasattr(snapshot, "tasks") and snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, "state") and task.state:
                sub_msg, sub_vals = _extract_interrupt(task.state)
                values = {**values, **sub_vals}
                if sub_msg: msg = sub_msg
            if not msg and hasattr(task, "interrupts") and task.interrupts:
                msg = str(task.interrupts[0].value)
    return msg, values


def _json_loads_safe(raw, fallback):
    if raw in (None, "", []):
        return fallback
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    return raw


def _build_state_payload(values: dict) -> dict:
    task_plan = _json_loads_safe(values.get("task_plan"), [])
    task_results = _json_loads_safe(values.get("task_results"), [])
    dataset_issues = _json_loads_safe(values.get("dataset_issues"), [])
    cleaning_strategy = _json_loads_safe(values.get("cleaning_strategy"), [])
    current_idx = values.get("current_task_index", 0) or 0
    current_title = values.get("current_task_title")
    next_step = values.get("next_step")

    if not current_title and task_plan and current_idx < len(task_plan):
        current_title = task_plan[current_idx].get("title")

    return {
        "working_files": values.get("working_files", {}),
        "task_plan": task_plan,
        "task_results": task_results,
        "current_task_index": current_idx,
        "current_task_title": current_title,
        "current_task_type": next_step,
        "active_agent": values.get("active_agent"),
        "pipeline_status": values.get("pipeline_status"),
        "dataset_issues": dataset_issues,
        "cleaning_strategy": cleaning_strategy,
        "charts_generated": values.get("charts_generated", []),
        "ml_report": values.get("ml_report"),
        "insights": values.get("insights", []),
        "agent_log": values.get("agent_log", []),
        "analysis_result": values.get("analysis_result"),
        "data_quality_score": values.get("data_quality_score"),
        "deep_profile_report": bool(values.get("deep_profile_report")),
    }


# ── Checkpointer ─────────────────────────────────────────────────────
_memory_saver = None

def _get_memory_saver():
    global _memory_saver
    if _memory_saver is None:
        from langgraph.checkpoint.memory import MemorySaver
        _memory_saver = MemorySaver()
        logger.info("Using MemorySaver (in-memory, no persistence)")
    return _memory_saver

async def _make_checkpointer(pool):
    if pool is None or not USE_POSTGRES:
        return _get_memory_saver(), None
    try:
        conn = await pool.getconn()
        await conn.set_autocommit(True)
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return AsyncPostgresSaver(conn), conn
    except Exception as e:
        logger.warning(f"Postgres checkpointer failed ({e}) — MemorySaver fallback")
        return _get_memory_saver(), None

@asynccontextmanager
async def _graph_context(pool):
    checkpointer, conn = await _make_checkpointer(pool)
    compiled = super_graph.compile(checkpointer=checkpointer)
    try:
        yield compiled
    finally:
        if conn is not None:
            try: await pool.putconn(conn)
            except Exception: pass


# ── Background pipeline task registry ────────────────────────────────
# Maps thread_id → asyncio.Task (running) or result dict (finished)
_pipeline_tasks:   dict[str, asyncio.Task] = {}
_pipeline_results: dict[str, dict]         = {}


def _make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler] if langfuse_handler else []}


async def _run_pipeline_bg(thread_id: str, inputs: dict, pool):
    """
    Run the LangGraph pipeline as a non-blocking background task.
    Writes final result to _pipeline_results[thread_id] when done.
    The SSE stream polls checkpoint state to emit live log entries.
    """
    try:
        async with _graph_context(pool) as compiled:
            config   = _make_config(thread_id)
            output   = await compiled.ainvoke(inputs, config)
            snapshot = await compiled.aget_state(config, subgraphs=True)
            snap_vals = snapshot.values if snapshot and snapshot.values else {}
            merged    = {**output, **snap_vals}
            payload   = _build_state_payload(merged)

            if snapshot.next:
                msg, _ = _extract_interrupt(snapshot)
                _pipeline_results[thread_id] = {
                    "status":           "paused",
                    "interrupt_msg":    msg,
                    **payload,
                }
            else:
                _pipeline_results[thread_id] = {
                    "status":           "success",
                    **payload,
                }
    except Exception as e:
        logger.error(f"[{thread_id}] Pipeline error", exc_info=True)
        _pipeline_results[thread_id] = {"status": "error", "message": str(e)[:500]}
    finally:
        _pipeline_tasks.pop(thread_id, None)


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = None
    if USE_POSTGRES:
        try:
            import psycopg
            from psycopg_pool import AsyncConnectionPool
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            async with await psycopg.AsyncConnection.connect(DB_URI, autocommit=True) as conn:
                await AsyncPostgresSaver(conn).setup()
            pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, open=False)
            await pool.open()
            app.state.pool = pool
            app.state.checkpointer = "postgres"
            logger.info("PostgreSQL ready")
        except Exception as e:
            logger.error(f"PostgreSQL failed: {e} — MemorySaver")
            app.state.pool = None
            app.state.checkpointer = "memory"
    else:
        app.state.pool = None
        app.state.checkpointer = "memory"

    async def _cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            try: cleanup_stale_users()
            except Exception as e: logger.error(f"Cleanup: {e}")

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    if pool:
        try: await pool.close()
        except Exception: pass
    if langfuse_handler:
        try: langfuse_handler.flush()
        except Exception: pass


app = FastAPI(title="Maya AI Data Analyst", version="3.2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

super_graph = build_super_graph()


# ── SSE Stream ────────────────────────────────────────────────────────
async def _sse_stream(thread_id: str, pool) -> AsyncGenerator[str, None]:
    """
    Real-time SSE stream.

    Polls checkpoint every 500ms for new log entries.
    When background task finishes, emits final event and closes.

    Event shapes:
      {type:"log",      entry:{agent,action,detail,status}}
      {type:"progress", stage:str, pct:int}
      {type:"done",     result:{status,working_files,...}}
      {type:"paused",   interrupt_msg:str, ...}
      {type:"error",    message:str}
    """
    max_polls      = 600  # 5 min at 500ms
    last_log_count = 0
    last_state_sig = None

    try:
        async with _graph_context(pool) as compiled:
            config = _make_config(thread_id)

            for _ in range(max_polls):
                await asyncio.sleep(0.5)

                result = _pipeline_results.get(thread_id)
                if result is not None:
                    # Flush any remaining log entries before closing
                    try:
                        snapshot = await compiled.aget_state(config, subgraphs=True)
                        if snapshot and snapshot.values:
                            state_payload = _build_state_payload(snapshot.values)
                            logs = state_payload.get("agent_log", [])
                            for entry in logs[last_log_count:]:
                                yield f"data: {_safe_json({'type': 'log', 'entry': entry})}\n\n"
                            yield f"data: {_safe_json({'type': 'state', **state_payload})}\n\n"
                    except Exception:
                        pass

                    if result["status"] == "paused":
                        yield f"data: {_safe_json({'type': 'paused', **result})}\n\n"
                    elif result["status"] == "error":
                        yield f"data: {_safe_json({'type': 'error', 'message': result.get('message','Unknown error')})}\n\n"
                    else:
                        yield f"data: {_safe_json({'type': 'done', 'result': result})}\n\n"
                    break

                # Pipeline still running — emit new log entries
                try:
                    snapshot = await compiled.aget_state(config, subgraphs=True)
                    if not snapshot or not snapshot.values:
                        continue

                    values = snapshot.values or {}
                    state_payload = _build_state_payload(values)
                    logs   = state_payload.get("agent_log", [])

                    for entry in logs[last_log_count:]:
                        yield f"data: {_safe_json({'type': 'log', 'entry': entry})}\n\n"

                    state_sig = _safe_json({
                        "task_plan": state_payload.get("task_plan", []),
                        "task_results": state_payload.get("task_results", []),
                        "current_task_index": state_payload.get("current_task_index", 0),
                        "current_task_title": state_payload.get("current_task_title"),
                        "active_agent": state_payload.get("active_agent"),
                        "pipeline_status": state_payload.get("pipeline_status"),
                        "dataset_issues": state_payload.get("dataset_issues", []),
                        "cleaning_strategy": state_payload.get("cleaning_strategy", []),
                    })
                    if state_sig != last_state_sig:
                        last_state_sig = state_sig
                        yield f"data: {_safe_json({'type': 'state', **state_payload})}\n\n"

                    if len(logs) > last_log_count:
                        last_log_count = len(logs)
                        # Emit progress tick
                        active = state_payload.get("active_agent", "") or ""
                        task_plan = state_payload.get("task_plan", [])
                        current_idx = state_payload.get("current_task_index", 0)
                        if task_plan:
                            pct = min(96, 10 + int((current_idx / max(len(task_plan), 1)) * 80))
                        else:
                            pct = 5
                        pct_map = {
                            "ingestion": 15, "merging": 28, "cleaning": 45,
                            "feature_engineering": 62, "ml": 80, "chat": 92, "planner": 10,
                        }
                        pct = max(pct, pct_map.get(active, 5))
                        yield f"data: {_safe_json({'type': 'progress', 'stage': active or 'running', 'pct': pct})}\n\n"

                except Exception as e:
                    yield f"data: {_safe_json({'type': 'error', 'message': str(e)})}\n\n"
                    break

    except Exception as e:
        yield f"data: {_safe_json({'type': 'error', 'message': str(e)})}\n\n"


# ══════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_files(
    thread_id:  str              = Form(...),
    user_id:    str              = Form(...),
    user_input: str              = Form("Load and analyze this data."),
    files:      list[UploadFile] = File(...),
):
    """Save files and fire the pipeline as a background task. Returns immediately."""
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Max {MAX_FILES} files per request")

    storage = get_storage(user_id)
    try:
        storage.check_quota()
    except QuotaExceededError as e:
        raise HTTPException(507, str(e))

    file_paths = []
    try:
        for file in files:
            raw  = await file.read()
            path = storage.save_upload(raw, file.filename)
            file_paths.append(path)
            logger.info(f"[{user_id}] Saved: {file.filename} → {path}")
    except Exception as e:
        raise HTTPException(500, f"File save error: {e}")

    _pipeline_results.pop(thread_id, None)

    inputs = {
        "file_paths":         file_paths,
        "working_files":      {},
        "messages":           [],
        "user_input":         user_input,
        "user_id":            user_id,
        "task_results":       [],
        "charts_generated":   [],
        "agent_log":          [],
        "insights":           [],
        "iteration_count":    0,
        "current_task_index": 0,
        "is_large_dataset":   False,
        "pipeline_status":    "running",
    }

    task = asyncio.create_task(_run_pipeline_bg(thread_id, inputs, app.state.pool))
    _pipeline_tasks[thread_id] = task

    return {
        "status":    "running",
        "thread_id": thread_id,
        "files":     [os.path.basename(p) for p in file_paths],
        "message":   f"Pipeline started — open /stream/{thread_id} for live updates.",
    }


@app.post("/resume")
async def resume_pipeline(
    thread_id: str = Form(...),
    user_id:   str = Form(...),
    feedback:  str = Form(...),
):
    """Resume a paused pipeline as a background task."""
    _pipeline_results.pop(thread_id, None)

    async def _resume_bg():
        try:
            async with _graph_context(app.state.pool) as compiled:
                config   = _make_config(thread_id)
                snapshot = await compiled.aget_state(config, subgraphs=True)
                if not snapshot or not snapshot.values:
                    _pipeline_results[thread_id] = {"status": "error", "message": "Thread not found"}
                    return

                if not snapshot.next:
                    snap_vals = snapshot.values or {}
                    msgs = snap_vals.get("messages", [])
                    payload = _build_state_payload(snap_vals)
                    _pipeline_results[thread_id] = {
                        "status":           "success",
                        "response":         msgs[-1].content if msgs else "Pipeline complete.",
                        **payload,
                    }
                    return

                output    = await compiled.ainvoke(Command(resume=feedback), config)
                snapshot2 = await compiled.aget_state(config, subgraphs=True)
                snap2     = snapshot2.values if snapshot2 and snapshot2.values else {}
                merged    = {**output, **snap2}
                payload   = _build_state_payload(merged)

                if snapshot2 and snapshot2.next:
                    msg, _ = _extract_interrupt(snapshot2)
                    _pipeline_results[thread_id] = {
                        "status":           "paused",
                        "interrupt_msg":    msg,
                        **payload,
                    }
                else:
                    msgs = merged.get("messages", [])
                    _pipeline_results[thread_id] = {
                        "status":           "success",
                        "response":         msgs[-1].content if msgs else "",
                        **payload,
                    }
        except Exception as e:
            logger.error("Resume error", exc_info=True)
            _pipeline_results[thread_id] = {"status": "error", "message": str(e)[:400]}
        finally:
            _pipeline_tasks.pop(thread_id, None)

    task = asyncio.create_task(_resume_bg())
    _pipeline_tasks[thread_id] = task
    return {"status": "running", "thread_id": thread_id}


@app.get("/stream/{thread_id}")
async def stream_activity(thread_id: str, request: Request):
    """SSE: real-time activity logs + final result event."""
    return StreamingResponse(
        _sse_stream(thread_id, app.state.pool),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pipeline-result/{thread_id}")
async def pipeline_result(thread_id: str):
    """Poll for background task result (fallback for clients that can't use SSE)."""
    if thread_id in _pipeline_tasks and not _pipeline_tasks[thread_id].done():
        return {"status": "running"}
    result = _pipeline_results.get(thread_id)
    return result if result else {"status": "not_found"}


@app.post("/chat")
async def chat(
    message:   str = Form(...),
    thread_id: str = Form(...),
    user_id:   str = Form(...),
):
    """Conversational queries — runs synchronously (short enough for one HTTP request)."""
    try:
        async with _graph_context(app.state.pool) as compiled:
            config         = _make_config(thread_id)
            current        = await compiled.aget_state(config)
            current_values = current.values if current and current.values else {}

            if current and current.next:
                msg, _ = _extract_interrupt(current)
                return {
                    "status": "paused",
                    "interrupt_msg": msg,
                    "thread_id": thread_id,
                    **_build_state_payload(current_values),
                }

            inputs = {
                **current_values,
                "messages":           list(current_values.get("messages", []))
                                      + [HumanMessage(content=message)],
                "user_input":         message,
                "user_id":            user_id,
                "iteration_count":    0,
                "current_task_index": current_values.get("current_task_index", 0),
                "charts_generated":   current_values.get("charts_generated", []),
                "agent_log":          current_values.get("agent_log", []),
                "insights":           current_values.get("insights", []),
            }

            output   = await compiled.ainvoke(inputs, config)
            snapshot = await compiled.aget_state(config, subgraphs=True)
            snap_vals = snapshot.values if snapshot and snapshot.values else output
            merged = {**output, **snap_vals}
            payload = _build_state_payload(merged)

            if snapshot.next:
                msg, _ = _extract_interrupt(snapshot)
                latest_vals = snapshot.values if snapshot and snapshot.values else merged
                return {
                    "status": "paused",
                    "interrupt_msg": msg,
                    "thread_id": thread_id,
                    **_build_state_payload(latest_vals),
                }

            msgs   = merged.get("messages", [])
            charts = payload.get("charts_generated", [])
            return {
                "status":    "success",
                "response":  msgs[-1].content if msgs else "No response generated.",
                "plot_path": charts[-1] if charts else None,
                **payload,
            }

    except Exception as e:
        logger.error("Chat error", exc_info=True)
        return JSONResponse(status_code=200, content={"status": "error", "message": str(e)[:500]})


@app.post("/plan")
async def start_plan(thread_id: str = Form(...), user_id: str = Form(...), user_input: str = Form(...)):
    """Run a multi-step planned pipeline as a background task."""
    _pipeline_results.pop(thread_id, None)

    async def _plan_bg():
        try:
            async with _graph_context(app.state.pool) as compiled:
                config         = _make_config(thread_id)
                current        = await compiled.aget_state(config)
                current_values = current.values if current and current.values else {}
                if not current_values.get("working_files"):
                    _pipeline_results[thread_id] = {"status": "error", "message": "No data loaded — upload files first."}
                    return
                inputs = {**current_values, "user_input": user_input, "user_id": user_id,
                          "iteration_count": 0, "messages": [],
                          "agent_log": current_values.get("agent_log", [])}
                output    = await compiled.ainvoke(inputs, config)
                snapshot  = await compiled.aget_state(config, subgraphs=True)
                snap_vals = snapshot.values if snapshot and snapshot.values else {}
                merged    = {**output, **snap_vals}
                payload   = _build_state_payload(merged)
                if snapshot.next:
                    msg, _ = _extract_interrupt(snapshot)
                    _pipeline_results[thread_id] = {
                        "status": "paused",
                        "interrupt_msg": msg,
                        **payload,
                    }
                else:
                    _pipeline_results[thread_id] = {
                        "status": "success",
                        **payload,
                    }
        except Exception as e:
            _pipeline_results[thread_id] = {"status": "error", "message": str(e)[:400]}
        finally:
            _pipeline_tasks.pop(thread_id, None)

    _pipeline_tasks[thread_id] = asyncio.create_task(_plan_bg())
    return {"status": "running", "thread_id": thread_id}


@app.get("/state/{thread_id}")
async def get_thread_state(thread_id: str):
    try:
        async with _graph_context(app.state.pool) as compiled:
            config   = _make_config(thread_id)
            snapshot = await compiled.aget_state(config, subgraphs=True)
            if not snapshot or not snapshot.values:
                return {"status": "not_found"}
            values        = snapshot.values
            state_payload = _build_state_payload(values)
            is_paused     = bool(snapshot.next)
            interrupt_msg = ""
            if is_paused:
                interrupt_msg, _ = _extract_interrupt(snapshot)
            return {
                "status":              "paused" if is_paused else "complete",
                "interrupt_msg":       interrupt_msg,
                **state_payload,
                "ml_model_path":       values.get("ml_model_path"),
            }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/statistics")
async def get_statistics(user_id: str):
    storage = get_storage(user_id)
    pickles = storage.list_pickles()
    if not pickles:
        raise HTTPException(404, "No data found")
    total_rows, total_cols, all_columns, sample_rows = 0, 0, {}, []
    try:
        for fname, fpath in pickles.items():
            df = pd.read_pickle(fpath)
            total_rows += len(df); total_cols += len(df.columns)
            for col in df.columns:
                key = f"[{fname}] {col}" if len(pickles) > 1 else col
                s   = df[col]
                cs  = {"dtype": str(s.dtype), "missing": int(s.isna().sum()),
                       "missing_pct": round(s.isna().mean()*100,1), "unique": int(s.nunique())}
                if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                    cs.update({"mean": float(s.mean()), "median": float(s.median()),
                               "std": float(s.std()), "min": float(s.min()), "max": float(s.max())})
                all_columns[key] = cs
            for rec in df.head(5).to_dict(orient="records"):
                sample_rows.append({(f"[{fname}] {k}" if len(pickles)>1 else k): v for k,v in rec.items()})
        return Response(
            content=json.dumps(_deep_clean({"total_rows": total_rows, "total_columns": total_cols,
                                            "files": list(pickles.keys()), "columns": all_columns,
                                            "sample_data": sample_rows[:20],
                                            "storage_mb": storage.storage_mb()}), cls=_NpEncoder),
            media_type="application/json")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/download")
async def download_data(user_id: str, filename: str = ""):
    storage = get_storage(user_id); pickles = storage.list_pickles()
    if not pickles: raise HTTPException(404, "No data found")
    fpath = pickles.get(filename) if filename else max(pickles.values(), key=lambda p: os.path.getsize(p))
    try:
        df = pd.read_pickle(fpath); csv_path = fpath.replace(".pkl","_export.csv")
        df.to_csv(csv_path, index=False)
        return FileResponse(csv_path, filename="maya_export.csv", media_type="text/csv")
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/download-forecast")
async def download_forecast(user_id: str):
    storage = get_storage(user_id); csvs = list(storage.models.glob("*_forecast.csv"))
    if not csvs: raise HTTPException(404, "No forecast CSV found")
    return FileResponse(str(max(csvs, key=lambda f: f.stat().st_mtime)), filename="forecast.csv", media_type="text/csv")

@app.get("/chart/{user_id}/{filename}")
async def serve_chart(user_id: str, filename: str):
    storage = get_storage(user_id)
    for p in [storage.charts / filename, storage.sandbox / filename]:
        if p.exists(): return FileResponse(str(p))
    raise HTTPException(404, "Chart not found")

@app.get("/model-report")
async def get_model_report(user_id: str):
    import pickle as pkl
    storage = get_storage(user_id); model_files = list(storage.models.glob("*.pkl"))
    if not model_files: raise HTTPException(404, "No trained model found")
    latest = max(model_files, key=lambda f: f.stat().st_mtime)
    try:
        with open(str(latest),"rb") as f: artifact = pkl.load(f)
        return {"model_type": type(artifact.get("model")).__name__,
                "task_type": artifact.get("task_type"), "target_col": artifact.get("target_col"),
                "feature_cols": artifact.get("feature_cols",[]),
                "feature_importance": artifact.get("feature_importance",{}),
                "metrics": artifact.get("metrics",{}), "model_path": str(latest)}
    except Exception as e: raise HTTPException(500, str(e))

@app.delete("/reset")
async def reset_user(user_id: str):
    for tid in list(_pipeline_tasks):
        _pipeline_tasks[tid].cancel()
    get_storage(user_id).cleanup()
    return {"status": "success"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.2",
            "checkpointer": getattr(app.state,"checkpointer","unknown"),
            "running_tasks": len(_pipeline_tasks), "timestamp": time.time()}

# ── Frontend static files ─────────────────────────────────────────────
for _d in [os.path.join(os.path.dirname(__file__),"..","frontend","dist"),
           os.path.join(os.path.dirname(__file__),"..","frontend")]:
    if os.path.exists(_d):
        app.mount("/", StaticFiles(directory=_d, html=True), name="frontend")
        break
