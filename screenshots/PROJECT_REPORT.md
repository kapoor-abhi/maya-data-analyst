# Maya (v3.2) — Detailed Project Report (Data Scientist + AI Engineer Focus)

**One-line pitch:** Maya is a multi-agent, production-oriented AI data analyst that ingests messy multi-file datasets, performs cleaning + feature engineering, trains ML models, and answers questions via SQL/analytics, with sandboxed execution, real-time observability, and resumable state.

**What this document is for:** A recruiter- and interview-ready technical report that demonstrates both (1) data-science thinking and (2) engineering depth: architecture, reliability, safety, and end-to-end delivery.

---

## 1) Outcome Summary (What This Project Delivers)

Maya provides an end-to-end “data analyst in a box” experience:

- **Upload multiple files** (CSV, Excel, Parquet, JSON).
- **Auto-ingest** into a standardized internal format (`.pkl` pickles per user session).
- **Merge** multiple tables using an LLM-suggested join strategy with human checkpoints + audit validation.
- **Clean** using a tool-driven ReAct loop that systematically profiles and fixes data quality issues while enforcing guardrails (e.g., no `dropna()` row loss).
- **Feature engineer** new predictive features and validate them (variance, correlation with target when available).
- **Train ML models** via a tool-based ML agent (AutoML style selection + cross-validation and hold-out evaluation), persist artifacts, and support forecasting + prediction.
- **Query & analyze** through a conversational “Chat” agent: Natural language to DuckDB SQL, visualization generation, advanced analytics (segmentation, clustering), statistical testing, and automated insights.
- **Run complex workflows** through a planner that generates a JSON task plan, requests approval, then dispatches tasks sequentially.
- **Operate reliably** with checkpointing (PostgreSQL-backed LangGraph checkpointer, with in-memory fallback) and real-time observability (SSE activity stream + structured logs).

The project outcome is not “a tool demo”; it is a **full system**: UI, API, multi-agent orchestration, persistence, sandbox execution, and a test suite that validates core functionality.

---

## 2) System Architecture (Minimal Diagram)

![Maya System Architecture](system_architecture.png)

Key design intent:

- **Safety:** LLM-generated code runs inside a Docker sandbox with `--network none` and resource/time limits.
- **Reliability:** Long pipelines run as background tasks; state is resumable via checkpoints.
- **Transparency:** A structured activity feed shows what each agent is doing in real time.
- **Data-science quality:** The agent workflow mirrors how a human DS works: profile → fix → verify → iterate → model → interpret.

---

## 3) End-to-End Runtime Flow (From a User’s Perspective)

### Upload + pipeline execution

1. User selects files in the UI.
2. UI opens SSE stream **before** starting upload (prevents missing early events).
3. UI `POST /upload` sends multipart form:
   - `thread_id`, `user_id`, `user_input`, and file bytes.
4. Backend persists uploads into per-user storage (`storage/<user_id>/sandbox/`) and starts a background LangGraph run.
5. UI consumes live SSE events:
   - `log` entries (structured agent activity)
   - `progress` ticks (stage + percent)
   - `paused` (human-in-the-loop checkpoint)
   - `done` (final result)
   - `error` (terminal failure)

### Pause + resume

When an agent requires human feedback (or hits repeated failures), the pipeline can **pause**. The UI sends:

- `POST /resume` with `thread_id`, `user_id`, and feedback text.

Because the graph uses checkpointing, the pipeline resumes exactly where it left off.

---

## 4) Core Engineering Design (AI Systems + Platform Thinking)

### 4.1 Unified State as the “Contract” Between Agents

Maya uses a single LangGraph state schema (`MasterState`) that all agents read and write. The state includes:

- conversational context (`messages`, `user_input`, `user_feedback`)
- data artifacts (`file_paths`, `working_files`)
- pipeline control (`error`, `iteration_count`, `next_step`, `pipeline_status`)
- outputs (`charts_generated`, `ml_report`, `insights`, `analysis_result`)
- operational metadata (`active_agent`, `agent_log`)

This is an engineering win because:

- Agents become **composable** and loosely coupled.
- The frontend can display progress without peeking into internal agent logic.
- Checkpointing becomes straightforward: checkpoint state is a single structured object.

### 4.2 Orchestration via a Super-Graph (Intent Routing + State-Dependent Dispatch)

The orchestrator (`core/super_agent.py`) is a LangGraph `StateGraph` that:

- routes requests via a **two-stage router**:
  - fast keyword heuristics for speed and determinism
  - LLM fallback router when ambiguous
- decides what to do next based on state:
  - if multiple files exist and unmerged, route to merge
  - if planner is active, keep returning control to the planner

This is a pragmatic, production-friendly hybrid: **heuristics for stability**, LLM routing for flexibility.

### 4.3 Non-Blocking Backend Execution + Real-Time SSE Observability

In `app/main.py`, pipeline execution is intentionally decoupled from HTTP request lifecycle:

- `POST /upload` stores files and starts an `asyncio.create_task(...)`.
- `GET /stream/{thread_id}` streams events:
  - it polls the live checkpoint state every ~500ms
  - emits incremental log entries and progress updates
  - emits a terminal event when the background task completes (success/paused/error)

This solves a common engineering challenge in agentic systems:

- LLM pipelines often take longer than typical HTTP timeouts.
- You want UI feedback while the pipeline runs.
- You need robust resume after process crash or API limits.

### 4.4 Checkpointing for Resume Capability (Postgres + Memory Fallback)

The backend compiles the super-graph with:

- **PostgreSQL-backed checkpointer** when available (`AsyncPostgresSaver`)
- **in-memory saver** fallback when Postgres is unavailable or misconfigured

This architecture allows:

- persistence across server restarts
- recovery from rate limits or transient failures
- clean separation between compute and state

### 4.5 Sandboxed Code Execution (Security + Reproducibility)

LLM-generated transformation code is executed through `core/sandbox.py`:

- Docker-based execution:
  - `--network none`
  - CPU and memory limits
  - timeout enforcement (`SANDBOX_TIMEOUT_SECS`)
  - mounted per-user storage (for pickles/models/charts)
- Local subprocess fallback exists for developer convenience but is explicitly flagged as unsafe for production.

This is central to making an “agent that writes code” defensible:

- The system expects to run arbitrary analysis scripts, but keeps the host protected.
- Artifacts are reproducible because the runtime environment is pinned via `Dockerfile.sandbox`.

### 4.6 Per-User Storage Isolation, Quotas, and Cleanup

`core/storage.py` implements a per-user directory layout:

- `storage/<user_id>/sandbox/` (uploaded + intermediate pickles)
- `storage/<user_id>/models/` (trained model artifacts)
- `storage/<user_id>/charts/` (generated charts)
- `storage/<user_id>/reports/` (future reporting output)

Engineering considerations addressed:

- **isolation:** prevents cross-user leakage
- **quotas:** `MAX_USER_STORAGE_MB`
- **eviction:** TTL cleanup for stale users and old files (`FILE_TTL_HOURS`)

### 4.7 Observability: Structured Logs, Optional Tracing, Optional Caching

- Every major action can be written as a structured activity entry (`core/activity_log.py`).
- The frontend renders these entries live as an activity feed.
- Optional:
  - Langfuse callback handler for tracing
  - Redis-backed LLM cache (when configured)

This is a critical engineering dimension: for agentic systems, debugging without observability is not sustainable.

---

## 5) Agent-by-Agent Deep Dive (Data Science Thinking Made Operational)

### 5.1 Ingestion Agent (File loading + memory optimization + safe transformations)

Responsibilities:

- Load raw files into pandas:
  - CSV: encoding + separator inference (`_smart_read_csv`)
  - Excel/Parquet/JSON supported
- Convert datasets into `.pkl` for faster downstream iteration
- Memory optimization for large datasets:
  - object → category when low cardinality
  - numeric downcasting
- Optional LLM-driven “column selection/transformation” stage:
  - strict guardrails: no row dropping and no datatype casting here (cleaning does that)
- Sandbox execution + rollback:
  - backups created before running LLM code
  - `.bak` rollback on failure

Engineering challenges solved:

- messy encodings and separators
- large-data memory pressure
- executing model-written code with rollback

### 5.2 Merge Agent (Join strategy suggestion + audit + self-correction)

Responsibilities:

- If multiple unmerged files exist, generate a merge strategy:
  - schema inspection (column lists, shapes, sample rows)
  - LLM suggestion emphasizes outer joins to reduce silent data loss
- Generate merge code in pandas and run in sandbox
- Audit validation:
  - fail if merged output has 0 rows (indicates wrong join keys)
- Self-correcting loop:
  - retry merge generation on errors
  - escalate to human checkpoint after repeated failures

This mirrors how a DS actually works: *“Try a join, inspect result, fix keys, avoid data loss.”*

### 5.3 Cleaning Agent (Tool-based ReAct loop + safety guardrails)

This is the most “data-scientist shaped” part of the system: it codifies a professional cleaning workflow.

Core behavior:

- A stable system prompt defines an explicit workflow:
  - `profile_dataframe()` first
  - then targeted inspections based on flags
  - then `run_transformation()` to fix
  - always `verify_result()` / `compare_before_after()` after changes
- Tools cover a wide checklist:
  - missingness, duplicates, ranges, outliers, categories normalization, datetime parsing, encoding issues
  - multicollinearity/VIF, scaling recommendations, sparsity analysis
  - PII scanning and masking recommendations
  - leakage risk scanning
  - class imbalance analysis (recommendations only, no SMOTE on full dataset)
- Human-in-the-loop can be triggered if a domain decision is needed.

Safety measures:

- A code-safety gate blocks suspicious primitives (`eval`, `exec`, subprocess, sockets, etc.).
- Strong transformation rules prevent destructive cleanup (e.g., no `dropna()`).
- Iteration caps prevent runaway loops (`MAX_TOOL_TURNS`).

Engineering challenge addressed:

LLMs are powerful but nondeterministic; a tool-driven loop + verification steps makes the cleaning process more auditable and more stable.

### 5.4 Feature Engineering Agent (Create features, validate, iterate)

Responsibilities:

- Inspects data for FE opportunities.
- Generates and executes FE code in sandbox.
- Validates that new columns exist and have variance; optionally checks correlation to target.
- ReAct loop continues until completion criteria are met (or max turns reached).
- Human checkpoint after FE for approval or additional requests.

This turns “feature engineering” from vague LLM output into a measurable process.

### 5.5 ML Agent (Auto-train + forecasting + custom ML code)

The ML agent is designed to support both “standard supervised learning” and advanced workflows.

Key capabilities:

- **Inspect for ML readiness:** identify candidate target columns, missingness patterns, and feature types.
- **Auto-train tool:**
  - feature selection: primarily numeric features, with basic categorical encoding
  - robust NaN/inf handling
  - model family selection:
    - XGBoost, LightGBM (when available)
    - RandomForest
    - linear models (Ridge / LogisticRegression)
    - gradient boosting fallback
  - evaluation:
    - cross-validation (KFold / StratifiedKFold / TimeSeriesSplit depending on setup)
    - explicit train/test hold-out for “honest” metrics
  - artifacts saved to `storage/<user_id>/models/*.pkl` with:
    - model object
    - feature list
    - metrics
    - feature importance or coefficient-based importance
  - forecasting mode:
    - generates a future-date table and writes a forecast CSV alongside the model artifact
- **Custom ML code tool:** for Prophet/ARIMA/LSTM/SVM/stacking or any non-standard approach.
- **Predict tool:** apply a saved model and append predictions back into the dataset.

Data-science thinking embedded:

- model evaluation separated into CV for selection + hold-out for sanity
- feature leakage avoidance begins earlier in cleaning (leakage scanning tool)
- interpretability via feature importance

### 5.6 Chat Agent (DuckDB SQL + visualization + analytics + statistics + insights)

This agent makes “question answering” operational and scalable:

- **Intent router** classifies user requests into:
  - `query` (SQL)
  - `visualize` (chart generation)
  - `analytics` (segmentation, anomaly detection, etc.)
  - `statistics` (hypothesis testing)
  - `insights` (auto findings)
  - `custom_code` (anything else)
- **DuckDB SQL path:**
  - registers the pandas DataFrames as DuckDB tables
  - generates SQL via LLM with schema grounding
  - auto-retries on SQL errors (up to 3)
  - summarizes results in business language
- **Visualization path:**
  - generates matplotlib/seaborn scripts
  - saves PNG to `storage/<user_id>/charts/`
  - returns the chart path so UI can render it
- **Analytics path:**
  - supports clustering / segmentation / anomaly detection with code generation
  - captures charts if created and summarizes output
- **Statistics path:**
  - LLM writes scipy-based test scripts and prints statistics/p-values
  - then LLM interprets significance in plain English
- **Insights path:**
  - generates 5-10 findings from a computed data summary

Engineering challenges addressed:

- SQL generation is error-prone; the retry loop provides robustness.
- Chart generation is sandboxed and returns only the saved path, keeping UI simple and safe.

### 5.7 Planner Agent (Human-Approved JSON Plans + Task Dispatch)

The planner provides “project management” for data work:

- Creates a JSON array plan with up to 12 atomic tasks, including types like:
  - ingest, merge, clean, feature_engineer
  - ml_train, ml_predict
  - analyze, visualize, statistical_test, insights, export, report
- Presents the plan to the user for approval or revision.
- Dispatches each task by setting `user_input` and `next_step`, then returning control to the super-agent.
- Records task outcomes for a final summary (success vs error).
- Escalates repeated failures to human review with options to skip or abort.

This is a strong “AI engineer” feature because it:

- constrains long workflows into structured steps
- improves predictability and auditability
- keeps LLM context bounded (clears messages between tasks)

---

## 6) Frontend Engineering (SSE-Driven, State-Aware UI)

The UI is intentionally simple but engineered for “agentic” interaction:

- **SSE-first execution model:** the UI opens SSE *before* upload so early logs are not lost.
- **Session restore:** thread id persists in localStorage; UI rehydrates from `GET /state/{thread_id}`.
- **Activity feed:** renders structured `agent_log` entries live, not after completion.
- **Tabs:** Chat, Data (column stats), Plan, Model.
- **Model view:** renders metrics + feature importance bars.
- **Chart rendering:** uses `GET /chart/{user_id}/{filename}` to serve images saved to storage.

---

## 7) Tools and Technology Stack (What Recruiters Expect to See)

### Backend / AI Orchestration

- Python 3.11 (Docker) / Python 3.10 (local venv present)
- FastAPI, Uvicorn
- LangChain + LangGraph (multi-agent state machine and checkpointing)
- Multi-provider LLM support via LangChain integrations:
  - Groq, OpenAI, Anthropic, Google Gemini, Ollama
- PostgreSQL (LangGraph checkpoints)
- Redis (optional LLM response caching)
- Optional Langfuse (observability / traces)

### Data / ML / Stats

- pandas, numpy
- DuckDB (SQL over in-memory DataFrames)
- scipy (hypothesis tests)
- scikit-learn (modeling + CV + metrics)
- xgboost, lightgbm (optional depending on environment)
- imbalanced-learn (imbalance tooling)

### Visualization

- matplotlib, seaborn
- plotly + kaleido (static export support)

### DevOps / Security

- Docker, Docker Compose
- Docker-based sandbox image for executing LLM-written code:
  - `--network none`, resource limits, timeouts

### Testing

- pytest-based unit tests (core modules, graphs compile, tools)
- integration test script for end-to-end API + SSE behavior

---

## 8) Engineering Challenges and How This Project Addresses Them

### Challenge: LLM nondeterminism and fragile code generation

Mitigations implemented:

- Tool schemas for ReAct loops (FE/ML/Cleaning).
- Iteration caps and retry logic on failures (SQL, charting, merge, custom analysis).
- Human-in-the-loop checkpoints with optional `AUTO_APPROVE` for CI/testing.

### Challenge: Safely executing LLM-generated code

Mitigations implemented:

- Docker sandbox with network disabled.
- Resource + timeout limits.
- Code fence stripping and minimal code injection patterns.
- In cleaning: explicit code-safety gate to block dangerous primitives.

### Challenge: Long-running pipelines and UI feedback

Mitigations implemented:

- Background tasks (avoid blocking HTTP handlers).
- SSE for real-time logs and progress.
- Checkpoint polling to stream updates even if agents run long.

### Challenge: Resume after restart / rate limit / crash

Mitigations implemented:

- PostgreSQL-backed checkpointing with fallback.
- `GET /state/{thread_id}` endpoint to rehydrate UI.
- `/resume` to continue with user feedback.

### Challenge: Multi-file data integration without silent errors

Mitigations implemented:

- LLM-suggested merge strategy plus audit check (no 0-row merges).
- Outer join default to reduce data loss.
- HITL escalation after repeated merge failures.

---

## 9) Data Scientist Thinking (How the System Mirrors Real DS Work)

This project is intentionally designed to mimic a senior DS workflow:

- **Problem framing:** planner creates an explicit task plan before execution.
- **Schema grounding:** every agent inspects schemas and uses them to constrain generated code.
- **EDA:** profiling tools summarize missingness, distribution, cardinality, encoding artifacts.
- **Data quality discipline:** cleaning enforces non-destructive rules (impute vs drop).
- **Leakage awareness:** dedicated tools and prompts call out leakage and temporal issues.
- **Modeling discipline:** hold-out metrics + CV-based selection, feature importance extraction.
- **Communication:** outputs are summarized in business-friendly language, with charts and an activity trail.

In interviews, you can position this as building a “DS copilot” that operationalizes:

- repeatable analysis patterns
- safe execution primitives
- auditability and traceability

---

## 10) Testing Strategy (Why This Is More Than a Prototype)

The repo contains:

- **Unit tests** (`tests/test_maya.py`) that cover:
  - core modules (storage, sandbox, activity log, state schema)
  - tool functions (DuckDB schema utilities, etc.)
  - graph compilation for all agents + super graph
- **Integration test** (`tests/api_integration_test.py`) that:
  - hits `/health`
  - uploads multiple messy CSVs
  - reads SSE stream live
  - validates stats endpoints
  - runs chat SQL queries, clustering, and chart generation
  - checks state persistence

This is a meaningful signal to recruiters: the system has been exercised end-to-end and is designed to be operated, not just shown.

---

## 11) Security and Operational Notes (Recruiter-Friendly Transparency)

### Sandboxing and safety

- The system is explicitly sandbox-first for generated code.
- Cleaning agent blocks several high-risk Python primitives.

### Secret management (important)

The repository currently contains a `.env` file with what appear to be real API keys. In a real production or portfolio setting:

- API keys should be removed from git history and rotated.
- `.env` should contain placeholders only and be ignored via `.gitignore`.
- Use a secret manager (or platform-specific env injection) for deployment.

Calling this out proactively in your report/interview is a positive signal: it shows security awareness and engineering maturity.

---

## 12) Known Limitations and Practical Next Steps

If we treat this as a product, the next engineering steps would be:

- Add authentication + per-user identity, not just a localStorage user id.
- Harden sandbox further (seccomp/AppArmor profiles, stricter filesystem mounts, allowlist imports).
- Implement a stricter policy engine for custom analysis code paths.
- Improve join strategy validation (row counts per key, duplicate key rates, referential integrity checks).
- Add dataset versioning and artifact lineage (data → features → model).
- Add real model registry semantics (metadata, training dataset hash, reproducibility).
- Add structured reporting output (HTML/PDF) to formalize deliverables.

---

## 13) How to Demo This to Recruiters (A Reliable Script)

1. Start services (Docker Compose).
2. Upload 3 related CSVs (customers/orders/products).
3. Use prompt:
   - “Run a full pipeline: profile issues, clean, engineer features, train a churn model, and give insights.”
4. Show:
   - activity feed (engineering transparency)
   - plan tab (structured DS thinking)
   - model tab (metrics + importance)
   - data tab (missingness + types)
5. Ask chat questions:
   - “What are the top drivers of churn?”
   - “Plot churn rate by age band.”
   - “Run a statistical test comparing spend for churn vs non-churn.”

This demonstrates both your DS judgment and your engineering system-building ability.

---

## Appendix A) API Surface (Backend Endpoints)

Primary endpoints implemented in `app/main.py`:

- `POST /upload` — Upload files and start pipeline as a background task.
- `GET /stream/{thread_id}` — SSE stream of live activity and final result.
- `POST /resume` — Resume a paused pipeline with user feedback.
- `POST /chat` — Conversational query execution (synchronous, short path).
- `POST /plan` — Run planner flow in background (requires data already loaded).
- `GET /state/{thread_id}` — Fetch checkpointed state (used for UI restore).
- `GET /statistics?user_id=...` — Column-level stats + sample rows for loaded datasets.
- `GET /download?user_id=...&filename=...` — Export a dataset to CSV.
- `GET /download-forecast?user_id=...` — Download forecast CSV produced by ML agent.
- `GET /chart/{user_id}/{filename}` — Serve chart images.
- `GET /model-report?user_id=...` — Read latest trained model artifact and return metrics/importance.
- `DELETE /reset?user_id=...` — Delete all user data and cancel tasks.
- `GET /health` — Health probe including checkpointer type and running task count.

---

## Appendix B) Repository Map (What Lives Where)

- `app/`
  - FastAPI server, SSE, background execution, checkpoint wiring
- `core/`
  - `state.py` — shared state contract
  - `super_agent.py` — orchestrator graph + intent router
  - `llm.py` — provider factory (Groq/OpenAI/Anthropic/Gemini/Ollama)
  - `sandbox.py` — docker-based secure execution
  - `storage.py` — per-user isolation + quota + cleanup
  - `activity_log.py` — structured log entries for UI
- `agents/`
  - `ingestion/` — file loaders + optimization + optional transformations
  - `merging/` — join planning + sandbox merge execution + audits
  - `preprocessing/` — cleaning ReAct tools + safe transformations + verification
  - `feature_engineering/` — FE ReAct loop + validation
  - `ml/` — AutoML + forecasting + custom ML scripts
  - `chat/` — DuckDB SQL + visualizations + analytics + stats + insights
  - `planning/` — task plan creation + approval + dispatch
- `frontend/`
  - single-page UI (EventSource SSE, file upload, state restore, plan/model render)
- `tests/`
  - unit tests and an end-to-end integration script
- `Dockerfile` and `docker-compose.yml`
  - deployment and local orchestration

---

## Appendix C) Configuration Knobs (Operational Controls)

The system is primarily configured via environment variables:

- `LLM_PROVIDER` and provider-specific API keys/model names.
- `DATABASE_URL` (Postgres checkpointing).
- `REDIS_URL` (optional caching).
- `STORAGE_BASE`, `MAX_USER_STORAGE_MB`, `FILE_TTL_HOURS` (data lifecycle controls).
- `MAX_FILES_PER_REQUEST` (upload safety).
- `SANDBOX_IMAGE`, `SANDBOX_TIMEOUT_SECS`, `SANDBOX_MEM_LIMIT`, `SANDBOX_LOCAL` (execution controls).
- `AUTO_APPROVE` (bypass HITL for test/CI).
- `ALLOWED_ORIGINS` (CORS policy).
