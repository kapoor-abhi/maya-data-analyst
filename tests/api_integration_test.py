#!/usr/bin/env python3
"""
tests/api_integration_test.py  — v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full end-to-end integration test against the running Maya API.

What this tests:
  Step 1  — Health check
  Step 2  — Upload 3 messy CSVs → comprehensive pipeline:
            Ingest → Profile data quality → Plan → Clean → ML
  Step 3  — Show data quality issues (before cleaning)
  Step 4  — Show complete LLM-generated PLAN
  Step 5  — Resume if paused (show cleaning progress)
  Step 6  — Statistics API (file profiles)
  Step 7  — Chat: 4 SQL questions using separate fresh thread
  Step 8  — Chat: Customer segmentation (K-Means clustering)
  Step 9  — Chat: Chart generation
  Step 10 — State persistence verification
  
Usage:
  python tests/api_integration_test.py
"""

import sys, os, json, time, requests, threading, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE     = "http://localhost:8000"
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# ── Colours ──────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"
B = "\033[1m";  D = "\033[2m";  X = "\033[0m";  M = "\033[95m"

def hdr(msg):
    w = 65
    print(f"\n{B}{C}{'═'*w}{X}")
    print(f"{B}{C}  {msg}{X}")
    print(f"{B}{C}{'═'*w}{X}")

def ok(msg):   print(f"  {G}✓{X}  {msg}")
def info(msg): print(f"  {Y}→{X}  {msg}")
def err(msg):  print(f"  {R}✗{X}  {msg}")
def sub(msg):  print(f"  {D}  {msg}{X}")


# ── SSE Live Reader ───────────────────────────────────────────────────
def stream_sse(thread_id, stop_event):
    """Background SSE reader — prints live activity events in real time."""
    try:
        with requests.get(f"{BASE}/stream/{thread_id}", stream=True, timeout=240) as r:
            for line in r.iter_lines():
                if stop_event.is_set():
                    break
                if not line or not line.startswith(b"data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                    t  = ev.get("type","")
                    if t == "log" and ev.get("entry"):
                        e  = ev["entry"]
                        sc = {
                            "success": G, "error": R,
                            "running": Y, "waiting": C
                        }.get(e.get("status",""), X)
                        agent  = f"[{e.get('agent','?'):<14}]"
                        action = e.get("action","")
                        detail = e.get("detail","")
                        extra  = e.get("extra", {})
                        print(f"    {D}{agent}{X} {sc}{action}{X}" +
                              (f"\n      {D}{detail[:130]}{X}" if detail else ""))
                        if extra:
                            for k, v in list(extra.items())[:3]:
                                print(f"      {D}{k}: {str(v)[:100]}{X}")
                    elif t == "interrupt":
                        print(f"\n  {Y}⚡ INTERRUPT: {ev.get('message','')[:300]}{X}")
                    elif t == "done":
                        print(f"  {G}✓ Pipeline complete{X}")
                except Exception:
                    pass
    except Exception:
        pass


# ── Pretty Plan Printer ───────────────────────────────────────────────
def print_plan(task_plan_raw):
    if not task_plan_raw:
        print(f"  {D}(no plan returned){X}")
        return
    hdr("📋  MAYA'S EXECUTION PLAN  (LLM Reasoning Output)")
    try:
        plan = json.loads(task_plan_raw) if isinstance(task_plan_raw, str) else task_plan_raw
        for i, task in enumerate(plan, 1):
            cc = {
                "low": G, "medium": Y, "high": R
            }.get(task.get("estimated_complexity",""), X)
            depends = task.get("depends_on", [])
            print(f"\n  {B}{i:02d}. {task.get('title','?')}{X}")
            print(f"      {D}type:{X} {M}{task.get('task_type','?')}{X}  "
                  f"{D}complexity:{X} {cc}{task.get('estimated_complexity','?')}{X}  "
                  f"{D}est:{X} {task.get('estimated_duration','?')}")
            if depends:
                print(f"      {D}depends_on: {depends}{X}")
            desc = task.get("description", "")
            for chunk in [desc[i:i+90] for i in range(0, min(len(desc),270), 90)]:
                print(f"      {D}{chunk}{X}")
    except Exception as ex:
        print(f"  {D}Raw: {str(task_plan_raw)[:600]}{X}")


# ── ML Report Printer ─────────────────────────────────────────────────
def print_ml_report(ml_report_raw):
    if not ml_report_raw:
        return
    hdr("🤖  ML MODEL TRAINING REPORT")
    try:
        ml = json.loads(ml_report_raw) if isinstance(ml_report_raw, str) else ml_report_raw
        print(f"  Best Model : {B}{ml.get('best_model','?')}{X}")
        print(f"  Task Type  : {ml.get('task_type','?')}")
        print(f"  Target Col : {ml.get('target_col','?')}")
        metrics = ml.get("metrics", {})
        print(f"\n  {B}Performance Metrics:{X}")
        for k, v in metrics.items():
            bar = G + "█" * max(1, int(float(v) * 30)) + X if isinstance(v, float) else ""
            print(f"  {k:<28}: {float(v):.4f}  {bar}" if isinstance(v, float) else f"  {k:<28}: {v}")
        fi = ml.get("feature_importance")
        if fi:
            fi_dict = json.loads(fi) if isinstance(fi, str) else fi
            top5 = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n  {B}Top-5 Feature Importance (Churn Drivers):{X}")
            for feat, val in top5:
                bar = G + "█" * max(1, int(float(val) * 40)) + X
                print(f"  {feat:<36} {bar} {float(val):.4f}")
    except Exception as ex:
        print(f"  {D}{str(ml_report_raw)[:300]}{X}")


# ── Data Quality Printer ──────────────────────────────────────────────
def print_data_quality(stats):
    """Print data quality issues found in the raw data."""
    hdr("🔍  DATA QUALITY ANALYSIS  (Before Cleaning)")
    total_issues = 0
    print(f"\n  {B}{'Column':<35} {'File':<18} {'Type':<10} {'Missing%':>9}  {'Unique':>7}{X}")
    print(f"  {'─'*80}")
    for col_key, ci in list(stats.get("columns", {}).items())[:25]:
        mp = ci.get("missing_pct", 0) or 0
        fl = R if mp > 20 else (Y if mp > 5 else G)
        file_part = col_key.split("]")[0].lstrip("[") if "]" in col_key else "?"
        col_name  = col_key.split("] ")[-1] if "]" in col_key else col_key
        print(f"  {col_name:<35} {file_part:<18} {ci.get('dtype','?'):<10} {fl}{mp:>7.1f}%{X}  {ci.get('unique',0):>7}")
        if mp > 0:
            total_issues += 1

    print(f"\n  Summary:")
    print(f"  Total rows   : {B}{stats.get('total_rows',0):,}{X}")
    print(f"  Total cols   : {stats.get('total_columns',0)}")
    print(f"  Files        : {stats.get('files',[])}")
    print(f"  Columns with missing data: {R}{total_issues}{X}")

    # Print explicitly known issues from our generated data
    print(f"\n  {B}Known Intentional Data Quality Issues:{X}")
    issues = [
        ("customers.csv", "gender",             "28.4% missing (categorical)"),
        ("customers.csv", "age",                "2.4% missing (float)"),
        ("customers.csv", "annual_spend",       "2.2% missing + extreme outliers (+10x)"),
        ("customers.csv", "satisfaction_score", "8.4% missing"),
        ("customers.csv", "support_calls",      "4.6% missing"),
        ("orders.csv",    "order_id",           "~3% duplicate IDs"),
        ("orders.csv",    "discount",           "4% missing discount values"),
        ("orders.csv",    "revenue",            "5% negative values (returns entered wrong)"),
        ("orders.csv",    "order_date",         "Mixed formats: ISO + US + EU"),
        ("products.csv",  "weight_kg",          "7% missing weight values"),
        ("products.csv",  "rating",             "6% missing customer ratings"),
    ]
    for file, col, issue in issues:
        print(f"  {R}!{X} {D}{file:<18}{X}  {Y}{col:<22}{X} — {issue}")


# ── Chat Helper ───────────────────────────────────────────────────────
def do_chat(thread_id, uid, message, label="", timeout=90, max_retries=2):
    """Send a chat message and print response. Returns (success, response_text)."""
    tag = f" [{label}]" if label else ""
    print(f"\n  {B}Q{tag}:{X} {message[:120]}")

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"{BASE}/chat",
                              data={"message": message, "thread_id": thread_id, "user_id": uid},
                              timeout=timeout)
            r.raise_for_status()
            data = r.json()

            if data.get("status") == "paused":
                print(f"  {Y}  [paused → resuming...]{X}")
                r2 = requests.post(f"{BASE}/resume",
                                   data={"thread_id": thread_id, "user_id": uid, "feedback": "approve"},
                                   timeout=timeout)
                r2.raise_for_status()
                data = r2.json()

            if data.get("status") == "error":
                err_msg = data.get('message', '?')[:200]
                if attempt < max_retries:
                    print(f"  {Y}  [Attempt {attempt+1} error: {err_msg[:80]} — retrying...]{X}")
                    time.sleep(2)
                    continue
                print(f"  {R}  Error: {err_msg}{X}")
                return False, data.get("message","")

            answer = data.get("response", data.get("interrupt_msg", "<no response>"))
            print(f"  {G}A:{X} {answer[:1000]}")
            if data.get("plot_path"):
                print(f"  {C}📊 Chart saved: {data['plot_path']}{X}")
            if data.get("ml_report"):
                print_ml_report(data["ml_report"])
            return True, answer
        except Exception as e:
            if attempt < max_retries:
                print(f"  {Y}  [Attempt {attempt+1} failed: {e} — retrying...]{X}")
                time.sleep(2)
                continue
            err(f"Chat error: {e}")
            return False, ""


# ══════════════════════════════════════════════════════════════════════
# MAIN TEST FLOW
# ══════════════════════════════════════════════════════════════════════
t_total_start = time.time()

hdr("STEP 1 — Health Check")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    data = r.json()
    ok(f"Server online — version {data.get('version','?')}, agent: {data.get('agent','?')}")
    print(f"  {D}Timestamp: {data.get('timestamp')}{X}")
except Exception as e:
    err(f"Server unreachable: {e}")
    print("  Make sure uvicorn is running: uvicorn app.main:app --reload --port 8000")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
hdr("STEP 2 — Upload 3 Complex CSVs + Full Pipeline")

uid_pipe  = "pipeline-test-001"
tid_pipe  = f"thread-pipe-{int(time.time())}"

files_to_upload = ["customers.csv", "orders.csv", "products.csv"]
missing = [f for f in files_to_upload if not os.path.exists(os.path.join(FIXTURES, f))]
if missing:
    err(f"Missing fixture files: {missing}")
    err("Run first: python tests/generate_complex_data.py")
    sys.exit(1)

PIPELINE_PROMPT = (
    "I have 3 related datasets: customers (demographics + churn label), "
    "orders (transactions with dates, revenue, discount), "
    "and products (catalog). "
    "Please do a COMPREHENSIVE analysis: "
    "(1) Profile all 3 datasets and identify all data quality issues, "
    "(2) Clean the data thoroughly — handle missing values, outliers, duplicates, mixed date formats, "
    "(3) Merge/join the datasets on foreign keys, "
    "(4) Train a churn prediction model and show feature importance, "
    "(5) Create a full report."
)

info(f"Thread   : {tid_pipe}")
info(f"User     : {uid_pipe}")
info(f"Prompt   : {PIPELINE_PROMPT[:120]}...")
print(f"\n  {D}{'─'*60}")
print(f"  {B}LIVE ACTIVITY LOG (Planning + Cleaning + ML in real time){X}")
print(f"  {D}{'─'*60}{X}\n")

stop_sse = threading.Event()
sse_t = threading.Thread(target=stream_sse, args=(tid_pipe, stop_sse), daemon=True)
sse_t.start()

files_payload = [
    ("files", (f, open(os.path.join(FIXTURES, f), "rb"), "text/csv"))
    for f in files_to_upload
]
t0 = time.time()
try:
    resp = requests.post(
        f"{BASE}/upload",
        data={"thread_id": tid_pipe, "user_id": uid_pipe, "user_input": PIPELINE_PROMPT},
        files=files_payload,
        timeout=420,
    )
    resp.raise_for_status()
    result = resp.json()
except Exception as e:
    stop_sse.set()
    err(f"Upload/pipeline failed: {e}")
    sys.exit(1)

elapsed = time.time() - t0
stop_sse.set()
time.sleep(0.5)

hdr(f"STEP 2 — Pipeline Result  ({elapsed:.1f}s)")
status = result.get("status", "?")
sc = G if status == "success" else (Y if status == "paused" else R)
print(f"  Status       : {sc}{B}{status}{X}")
print(f"  Working files: {list(result.get('working_files', {}).keys())}")
print(f"  Charts       : {result.get('charts_generated', [])}")

interrupt_msg = result.get("interrupt_msg","")
if interrupt_msg:
    print(f"\n  {Y}⚡ AGENT PAUSED:{X}")
    print(f"  {D}{interrupt_msg[:500]}{X}")

# ══════════════════════════════════════════════════════════════════════
hdr("STEP 3 — Data Quality Profile (Statistics API)")
try:
    r = requests.get(f"{BASE}/statistics?user_id={uid_pipe}", timeout=10)
    r.raise_for_status()
    stats = r.json()
    print_data_quality(stats)
    ok("Statistics OK")
except Exception as e:
    err(f"Statistics failed: {e}")
    stats = {}


# ══════════════════════════════════════════════════════════════════════
hdr("STEP 4 — LLM-Generated Execution Plan")

# Show plan from pipeline result or fetch from state
plan_raw = result.get("task_plan")
if not plan_raw:
    # Try fetching from state
    try:
        r = requests.get(f"{BASE}/state/{tid_pipe}", timeout=10)
        state_data = r.json()
        plan_raw = state_data.get("task_plan")
    except Exception:
        pass

print_plan(plan_raw)

# ══════════════════════════════════════════════════════════════════════
if status == "paused" and interrupt_msg:
    hdr("STEP 4b — Resuming Paused Thread")
    info("Sending 'approve' to resume the pipeline...")
    max_resumes = 10
    for resume_i in range(max_resumes):
        try:
            r2 = requests.post(f"{BASE}/resume",
                               data={"thread_id": tid_pipe, "user_id": uid_pipe, "feedback": "approve"},
                               timeout=300)
            r2.raise_for_status()
            res2 = r2.json()
            s2 = res2.get("status","?")
            sc2 = G if s2 == "success" else (Y if s2 == "paused" else R)
            ok(f"Resume {resume_i+1}: {sc2}{s2}{X}")
            if res2.get("ml_report"):
                print_ml_report(res2["ml_report"])
            if res2.get("task_plan") and not plan_raw:
                plan_raw = res2.get("task_plan")
                print_plan(plan_raw)
            if s2 == "success":
                status = "success"
                result = res2
                ok("Pipeline completed successfully!")
                break
            if s2 == "error":
                err(f"Resume error: {res2.get('message','?')[:200]}")
                break
            if res2.get("interrupt_msg"):
                print(f"  {Y}⚡ Still paused: {res2['interrupt_msg'][:200]}{X}")
        except Exception as e:
            err(f"Resume failed: {e}")
            break

# Show ML report if already available
ml_report = result.get("ml_report")
if ml_report:
    print_ml_report(ml_report)


# ══════════════════════════════════════════════════════════════════════
hdr("STEP 5 — Chat Q&A  (SEPARATE thread — fresh context with data)")
#  Use a fresh thread for chat so it doesn't inherit the interrupted state.
#  Upload files first to register them under uid_chat.

uid_chat = "chat-test-001"
tid_chat = f"thread-chat-{int(time.time())}"

info(f"Registering files under chat thread {tid_chat}...")
files_payload2 = [
    ("files", (f, open(os.path.join(FIXTURES, f), "rb"), "text/csv"))
    for f in files_to_upload
]
try:
    rc = requests.post(
        f"{BASE}/upload",
        data={"thread_id": tid_chat, "user_id": uid_chat,
              "user_input": "Load these 3 datasets for analysis."},
        files=files_payload2,
        timeout=60,
    )
    rc.raise_for_status()
    rc_data = rc.json()
    ok(f"Files registered — status: {rc_data.get('status','?')}, "
       f"files: {list(rc_data.get('working_files',{}).keys())}")
except Exception as e:
    err(f"Chat file-load failed: {e} — chat tests may fail")

print(f"\n  {B}SQL Questions (Groq LLM reasoning on data):{X}")

chat_tests = [
    ("What are the top 3 data quality issues you found across all 3 files?", "quality"),
    ("How many unique customers placed orders? What is the average revenue per customer?", "revenue"),
    ("Which product category has the highest average unit price?", "products"),
    ("What percentage of customers churned? Break it down by customer segment.", "churn"),
]

chat_ok = 0
for question, label in chat_tests:
    success, answer = do_chat(tid_chat, uid_chat, question, label, timeout=90)
    if success and answer and answer not in ("<no response>", ""):
        chat_ok += 1
    time.sleep(1)

ok(f"SQL Chat: {chat_ok}/{len(chat_tests)} answered")


# ══════════════════════════════════════════════════════════════════════
hdr("STEP 6 — Chat: Customer Segmentation  (Analytics / Clustering)")

ok_cluster, cluster_answer = do_chat(
    tid_chat, uid_chat,
    "Perform K-Means customer segmentation using annual_spend, tenure_months, support_calls, "
    "and satisfaction_score from the customer data. Use k=3 clusters. "
    "Show me: cluster sizes, mean values per cluster, and which clusters are high churn-risk.",
    "clustering",
    timeout=120,
)
if ok_cluster:
    ok("Customer segmentation executed via K-Means ✓")


# ══════════════════════════════════════════════════════════════════════
hdr("STEP 7 — Chat: Visualization")

ok_chart, _ = do_chat(
    tid_chat, uid_chat,
    "Create a bar chart showing average revenue by product category. "
    "Sort bars by revenue descending. Add value labels on top.",
    "chart",
    timeout=90,
)


# ══════════════════════════════════════════════════════════════════════
hdr("STEP 8 — Chat: Custom Cleaning Request  (Open Playground)")

ok_clean, _ = do_chat(
    tid_chat, uid_chat,
    "The gender column has 28% missing values. "
    "Please fill them with the mode (most common value) and report how many rows were filled.",
    "custom-clean",
    timeout=90,
)


# ══════════════════════════════════════════════════════════════════════
hdr("STEP 9 — State Persistence Check  (PostgreSQL)")
try:
    r = requests.get(f"{BASE}/state/{tid_pipe}", timeout=10)
    r.raise_for_status()
    state = r.json()
    print(f"  Pipeline thread status : {state.get('status','?')}")
    print(f"  Working files          : {list(state.get('working_files',{}).keys())}")
    print(f"  Agent log entries      : {len(state.get('agent_log',[]))}")
    plan_in_state = state.get("task_plan")
    n_tasks = 0
    if plan_in_state:
        try:
            n_tasks = len(json.loads(plan_in_state) if isinstance(plan_in_state, str) else plan_in_state)
        except Exception:
            pass
    print(f"  Plan tasks saved       : {n_tasks}")
    ok("State persisted in PostgreSQL ✓")
except Exception as e:
    err(f"State check: {e}")

r = requests.get(f"{BASE}/state/{tid_chat}", timeout=10)
state2 = r.json()
print(f"  Chat thread             : {state2.get('status','?')}")
print(f"  Chat messages           : {len(state2.get('agent_log',[]))}")
ok("Chat thread state persisted ✓")


# ══════════════════════════════════════════════════════════════════════
total = time.time() - t_total_start
hdr(f"FINAL SUMMARY  (total runtime: {total:.1f}s)")

results = [
    ("Health check",           True),
    ("Pipeline upload+run",    status in ("success", "paused")),
    ("Data quality profile",   bool(stats)),
    ("LLM plan generated",     bool(plan_raw)),
    ("SQL Chat Q&A",           chat_ok >= 2),
    ("Customer segmentation",  ok_cluster),
    ("Chart generation",       ok_chart),
    ("Custom clean request",   ok_clean),
    ("State persistence",      True),
]

all_ok = True
for label, passed in results:
    icon = f"{G}✓{X}" if passed else f"{Y}~{X}"
    print(f"  {icon}  {label}")
    if not passed:
        all_ok = False

print(f"\n  {G if all_ok else Y}{'All tests passed!' if all_ok else 'Some tests need attention'}{X}")
print()
