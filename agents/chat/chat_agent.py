"""
agents/chat/chat_agent.py — Conversational Data Analysis Agent

Full analyst capabilities:
- SQL queries via DuckDB (with retry on error)
- Chart generation (matplotlib/seaborn/plotly)
- Advanced analytics (clustering, segmentation, cohort, anomaly detection)
- Statistical tests (t-test, ANOVA, chi-square, correlation via scipy)
- Auto business insights (auto-summarize key patterns)
- Custom analysis code (anything the user asks for)

Intent router: query | visualize | analytics | statistics | insights | custom_code
"""
import os
import re
import json
import logging
import pandas as pd
import duckdb
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from core.state import MasterState
from core.llm import get_llm
from core.sandbox import DockerREPL, _strip_code
from core.activity_log import make_log_entry

logger = logging.getLogger(__name__)

router_llm   = get_llm("fast",  temperature=0.0)
coder_llm    = get_llm("coder", temperature=0.1)
repl_sandbox = DockerREPL()


def _get_schema(working_files: dict) -> tuple[str, dict]:
    """Returns text schema and clean DuckDB-ready DataFrames."""
    schema_info = ""
    clean_dfs   = {}
    for filename, path in working_files.items():
        table_name = re.sub(r'\W|^(?=\d)', '_', filename.replace('.pkl', '').replace('.csv', ''))
        try:
            df = pd.read_pickle(path)
            clean_dfs[table_name] = df
            schema_info += f"\nTable: `{table_name}`\n"
            for col, dtype in df.dtypes.items():
                n_unique = df[col].nunique()
                null_pct = round(df[col].isna().mean() * 100, 1)
                schema_info += f"  {col} ({dtype}) — {n_unique} unique, {null_pct}% null\n"
            schema_info += f"Rows: {len(df)}\n"
            schema_info += f"Sample: {df.head(2).to_dict(orient='records')}\n"
        except Exception:
            pass
    return schema_info, clean_dfs


def _get_data_summary(working_files: dict) -> str:
    """Quick statistical summary of all files for insight generation."""
    summary = ""
    for filename, path in working_files.items():
        try:
            df = pd.read_pickle(path)
            summary += f"\n=== {filename} ({len(df)} rows × {len(df.columns)} cols) ===\n"
            # Numeric summary
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                summary += "Numeric columns:\n"
                for col in num_cols[:10]:
                    s = df[col].dropna()
                    if len(s) > 0:
                        summary += (
                            f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
                            f"min={s.min():.2f}, max={s.max():.2f}, "
                            f"null={df[col].isna().sum()}\n"
                        )
            # Categorical summary
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                summary += "Categorical columns:\n"
                for col in cat_cols[:5]:
                    top = df[col].value_counts().head(3).to_dict()
                    summary += f"  {col}: top={top}, nunique={df[col].nunique()}\n"
            # Datetime summary
            dt_cols = df.select_dtypes(include="datetime64").columns.tolist()
            if dt_cols:
                for col in dt_cols[:2]:
                    summary += (f"  {col}: from {df[col].min()} to {df[col].max()}, "
                                f"span={(df[col].max() - df[col].min()).days} days\n")
        except Exception as e:
            summary += f"  [Error reading {filename}: {e}]\n"
    return summary


# ── Nodes ─────────────────────────────────────────────────────────────

def intent_router_node(state: MasterState) -> dict:
    """Route user request to the right analysis node."""
    user_input = state.get("user_input", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the user's data request as EXACTLY ONE word:

- "query"       — calculations, counts, averages, summaries, statistics, table results, comparisons, "how many", "what is the total"
- "visualize"   — charts, plots, graphs, heatmaps, distributions, trends, bar chart, line chart, scatter
- "analytics"   — clustering, segmentation, k-means, PCA, cohort analysis, RFM, anomaly detection, association rules, dimensionality reduction
- "statistics"  — t-test, ANOVA, chi-square, correlation test, hypothesis test, significance, p-value, statistical test
- "insights"    — auto-generate insights, key findings, what stands out, summarize patterns, business insights, EDA summary
- "custom_code" — anything else that needs custom Python analysis, complex multi-step analysis, or a specific algorithm

Respond with EXACTLY ONE WORD."""),
        ("user", "{request}"),
    ])

    response = (prompt | router_llm).invoke({"request": user_input})
    intent   = response.content.lower().strip().rstrip(".'\"")

    # Validate and map
    viz_words       = {"visualize", "plot", "chart", "graph", "bar", "line", "scatter",
                       "histogram", "heatmap", "distribution"}
    analytics_words = {"analytics", "cluster", "segment", "cohort", "pca", "anomaly",
                       "rfm", "k-means", "association"}
    stat_words      = {"statistics", "t-test", "anova", "chi-square", "correlation",
                       "hypothesis", "significance", "p-value", "statistical"}
    insight_words   = {"insights", "insight", "findings", "patterns", "summary", "eda"}

    if any(w in intent for w in stat_words):
        intent = "statistics"
    elif any(w in intent for w in insight_words):
        intent = "insights"
    elif any(w in intent for w in analytics_words):
        intent = "analytics"
    elif any(w in intent for w in viz_words):
        intent = "visualize"
    elif intent not in {"query", "visualize", "analytics", "statistics", "insights", "custom_code"}:
        intent = "query"

    log_entries = list(state.get("agent_log", []))
    log_entries.append(make_log_entry(
        "Chat", f"Intent: {intent}", user_input[:100], "running",
    ))
    return {"next_step": intent, "error": None, "iteration_count": 0,
            "agent_log": log_entries}


def sql_query_node(state: MasterState) -> dict:
    """Execute SQL query via DuckDB with auto-retry."""
    working_files = state.get("working_files", {})
    schema_info, clean_dfs = _get_schema(working_files)
    messages      = state.get("messages", [])
    context_msgs  = messages[-6:] if messages else []

    error_context = (
        f"\n\nFIX THIS SQL ERROR:\n{state.get('error')}\nQuery that failed: {state.get('python_code', '')}"
        if state.get("error") else ""
    )

    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert DuckDB SQL analyst.
Write a SQL query to answer the user's question using these tables.

Schema:
{schema}

Rules:
- Use DuckDB syntax (supports QUALIFY, STRFTIME, DATE_TRUNC, window functions, PIVOT, UNPIVOT)
- LIMIT results to 100 rows unless user asks for more
- For aggregations, use proper GROUP BY
- Column names with spaces must be quoted: "column name"
- Return ONLY valid SQL inside ```sql ... ``` blocks
{error_context}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    sql_response = (sql_prompt | coder_llm).invoke({
        "schema": schema_info,
        "messages": context_msgs,
        "error_context": error_context,
    })

    match     = re.search(r"```sql(.*?)```", sql_response.content, re.DOTALL | re.IGNORECASE)
    sql_query = match.group(1).strip() if match else sql_response.content.strip()

    try:
        conn = duckdb.connect()
        for table_name, df in clean_dfs.items():
            conn.register(table_name, df)
        result_df  = conn.execute(sql_query).df()
        result_str = result_df.head(50).to_string()
        conn.close()
    except Exception as e:
        current_iter = state.get("iteration_count", 0) + 1
        if current_iter > 3:
            return {
                "messages":        [AIMessage(content=f"SQL failed after 3 attempts: {e}")],
                "error":           str(e),
                "iteration_count": current_iter,
                "next_step":       "done",
            }
        return {
            "error":           f"SQL failed: {e}\nQuery: {sql_query}",
            "python_code":     sql_query,
            "iteration_count": current_iter,
            "next_step":       "query",
        }

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful data analyst. Summarize SQL results in clear, business-friendly language.
Include key numbers, trends, and actionable insights. Use markdown formatting.

SQL Query: {sql}
SQL Result ({n} rows):
{result}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    summary = (summary_prompt | coder_llm).invoke({
        "result": result_str, "n": len(result_df),
        "sql": sql_query, "messages": context_msgs,
    })

    log_entries = list(state.get("agent_log", []))
    log_entries.append(make_log_entry(
        "Chat", "Query complete",
        f"Returned {len(result_df)} rows", "success",
        {"rows": len(result_df), "sql": sql_query[:200]},
    ))
    return {
        "messages":        [AIMessage(content=summary.content)],
        "analysis_result": summary.content,
        "error":           None,
        "iteration_count": 0,
        "agent_log":       log_entries,
        "next_step":       "done",
    }


def visualizer_node(state: MasterState) -> dict:
    """Generate matplotlib/seaborn/plotly chart and return file path."""
    working_files = state.get("working_files", {})
    user_id       = state.get("user_id", "default")
    schema_info, _ = _get_schema(working_files)
    messages      = state.get("messages", [])
    context_msgs  = messages[-6:] if messages else []

    charts_dir = os.path.join("storage", user_id, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    error_feedback = (
        f"\n\nFIX THIS ERROR FROM PREVIOUS ATTEMPT:\n{state.get('error')}"
        if state.get("error") else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python data visualization expert.
Write a self-contained visualization script.

RULES:
1. Import pandas, matplotlib.pyplot as plt, seaborn as sns (as needed), uuid
2. working_files dict is pre-defined — load with pd.read_pickle(working_files[name])
3. Use actual column names from the schema below
4. Figure size: plt.figure(figsize=(12, 6)) or appropriate for chart type
5. Apply modern styling: plt.style.use('seaborn-v0_8') or sns.set_theme()
6. Save to: chart_path = f"{charts_dir}/chart_{{uuid.uuid4().hex[:8]}}.png"
7. plt.savefig(chart_path, dpi=150, bbox_inches='tight')
8. Print ONLY this line: print(f"SAVED_CHART:{{chart_path}}")
9. plt.close('all')
10. Add proper titles, axis labels, and legends

For time series: use plt.plot() with date on x-axis
For distributions: use sns.histplot() or sns.violinplot()
For correlations: use sns.heatmap(df.corr(), annot=True)
For categories: use sns.barplot() or sns.boxplot()

Schema: {schema}
Charts dir: {charts_dir}
Working Files: {files}
{error_feedback}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema_info,
        "charts_dir": charts_dir,
        "error_feedback": error_feedback,
        "messages": context_msgs,
    })

    code   = _strip_code(response.content)
    result = repl_sandbox.run(code)

    if result.get("error"):
        current_iter = state.get("iteration_count", 0) + 1
        if current_iter > 3:
            return {
                "messages":        [AIMessage(content=f"Chart failed after 3 attempts: {result['error']}")],
                "error":           result["error"],
                "iteration_count": current_iter,
                "next_step":       "done",
            }
        return {"error": result["error"], "iteration_count": current_iter, "next_step": "visualize"}

    chart_match      = re.search(r"SAVED_CHART:(.+\.png)", result.get("output", ""))
    charts_generated = list(state.get("charts_generated", []))
    log_entries      = list(state.get("agent_log", []))

    if chart_match:
        chart_path = chart_match.group(1).strip()
        charts_generated.append(chart_path)
        log_entries.append(make_log_entry("Chat", "Chart generated", chart_path, "success"))
        return {
            "messages":        [AIMessage(content="✅ Chart generated successfully.")],
            "charts_generated": charts_generated,
            "error":           None,
            "iteration_count": 0,
            "agent_log":       log_entries,
            "next_step":       "done",
        }

    return {
        "messages":        [AIMessage(content="Chart code ran but no file was saved.")],
        "charts_generated": charts_generated,
        "error":           None,
        "iteration_count": 0,
        "next_step":       "done",
    }


def analytics_node(state: MasterState) -> dict:
    """Run complex analytics (clustering, segmentation, cohort, anomaly detection) via sandbox."""
    working_files  = state.get("working_files", {})
    user_id        = state.get("user_id", "default")
    schema_info, _ = _get_schema(working_files)
    messages       = state.get("messages", [])
    context_msgs   = messages[-6:] if messages else []

    charts_dir = os.path.join("storage", user_id, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    error_feedback = (
        f"\n\nFIX THIS ERROR:\n{state.get('error')}" if state.get("error") else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python data scientist. Write a complete, self-contained analysis script.

EXECUTION RULES:
1. Import all libraries (pandas, numpy, sklearn, matplotlib, seaborn, scipy, etc.)
2. working_files dict is pre-defined: load with pd.read_pickle(working_files[name])
3. ALWAYS handle missing values BEFORE any ML: df.dropna() or SimpleImputer
4. ALWAYS encode categoricals BEFORE sklearn: LabelEncoder or pd.get_dummies
5. If creating charts: save to charts_dir with uuid name, print SAVED_CHART:<path>
6. Print key results clearly (cluster sizes, centroids, top segments, anomaly counts)
7. Do NOT use plt.show()

SUPPORTED ANALYSES:
- Clustering: KMeans, DBSCAN, AgglomerativeClustering (sklearn)
- Segmentation: RFM analysis, customer tiers
- Cohort analysis: retention rates over time
- Anomaly detection: IsolationForest, LocalOutlierFactor
- Association rules: mlxtend Apriori (if available)
- PCA / dimensionality reduction: sklearn PCA + visualization
- Time series decomposition: statsmodels seasonal_decompose

Schema: {schema}
Working Files (pre-defined as working_files dict): {files}
Charts dir (pre-defined as charts_dir): {charts_dir}
{error_feedback}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema_info,
        "charts_dir": charts_dir,
        "error_feedback": error_feedback,
        "messages": context_msgs,
    })

    # Inject working_files and charts_dir into code context
    preamble = (
        f"import os\n"
        f"working_files = {json.dumps(working_files)}\n"
        f"charts_dir = '{charts_dir}'\n"
        f"os.makedirs(charts_dir, exist_ok=True)\n"
    )
    code   = _strip_code(response.content)
    result = repl_sandbox.run(preamble + code)

    log_entries      = list(state.get("agent_log", []))
    charts_generated = list(state.get("charts_generated", []))

    if result.get("error"):
        current_iter = state.get("iteration_count", 0) + 1
        if current_iter > 3:
            log_entries.append(make_log_entry(
                "Chat", "Analytics failed", result["error"][:200], "error",
            ))
            return {
                "messages":        [AIMessage(content=f"Analytics failed after 3 attempts:\n{result['error'][:300]}")],
                "error":           result["error"],
                "iteration_count": current_iter,
                "agent_log":       log_entries,
                "next_step":       "done",
            }
        return {"error": result["error"], "iteration_count": current_iter,
                "next_step": "analytics"}

    output_text = result.get("output", "<no output>")
    for m in re.finditer(r"SAVED_CHART:(.+\.png)", output_text):
        charts_generated.append(m.group(1).strip())

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data scientist. Interpret this analytics output in clear, business-friendly language.
Include: key findings, cluster/segment characteristics, actionable recommendations.
Format with markdown headings and bullet points.

Raw Output:
{output}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    summary = (coder_llm).invoke(
        summary_prompt.format_messages(output=output_text[:3000], messages=context_msgs)
    )

    log_entries.append(make_log_entry(
        "Chat", "Analytics complete", output_text[:100], "success",
    ))
    return {
        "messages":        [AIMessage(content=summary.content)],
        "charts_generated": charts_generated,
        "analysis_result": summary.content,
        "error":           None,
        "iteration_count": 0,
        "agent_log":       log_entries,
        "next_step":       "done",
    }


def statistics_node(state: MasterState) -> dict:
    """Run statistical tests (t-test, ANOVA, chi-square, correlation) via scipy."""
    working_files  = state.get("working_files", {})
    user_id        = state.get("user_id", "default")
    schema_info, _ = _get_schema(working_files)
    messages       = state.get("messages", [])
    context_msgs   = messages[-6:] if messages else []

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a statistician. Write Python code to run the requested statistical tests.

AVAILABLE TESTS (use scipy.stats):
- t-test (independent/paired): scipy.stats.ttest_ind, ttest_rel
- ANOVA (one-way): scipy.stats.f_oneway
- Chi-square: scipy.stats.chi2_contingency
- Correlation: df.corr(), scipy.stats.pearsonr, spearmanr
- Normality: scipy.stats.shapiro
- Mann-Whitney U: scipy.stats.mannwhitneyu (non-parametric alternative to t-test)
- Kruskal-Wallis: scipy.stats.kruskal (non-parametric ANOVA)

RULES:
1. Import pandas, numpy, scipy.stats
2. working_files dict is pre-defined
3. Load data: pd.read_pickle(working_files[name])
4. Handle missing values before tests
5. Print test name, statistic, p-value, and interpretation (significant if p < 0.05)
6. Print correlation matrix if requested
7. Format output clearly

Schema: {schema}
Working Files (pre-defined): {files}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema_info,
        "messages": context_msgs,
    })

    preamble = (
        f"import os\n"
        f"working_files = {json.dumps(working_files)}\n"
    )
    code   = _strip_code(response.content)
    result = repl_sandbox.run(preamble + code)

    log_entries = list(state.get("agent_log", []))

    if result.get("error"):
        return {
            "messages":   [AIMessage(content=f"Statistical test failed: {result['error']}")],
            "error":      result["error"],
            "next_step":  "done",
            "agent_log":  log_entries,
        }

    output_text = result.get("output", "<no output>")

    # Interpret results with LLM
    interpret_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a statistician. Interpret these statistical test results in plain English.
Explain: what was tested, whether results are significant (p < 0.05), and what it means for the business.
Use markdown formatting.

Raw Test Output:
{output}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    interpretation = coder_llm.invoke(
        interpret_prompt.format_messages(output=output_text[:2000], messages=context_msgs)
    )

    log_entries.append(make_log_entry(
        "Chat", "Statistical tests complete", output_text[:200], "success",
    ))
    return {
        "messages":          [AIMessage(content=interpretation.content)],
        "statistical_tests": output_text,
        "analysis_result":   interpretation.content,
        "error":             None,
        "agent_log":         log_entries,
        "next_step":         "done",
    }


def insights_node(state: MasterState) -> dict:
    """Auto-generate business insights from data summary."""
    working_files = state.get("working_files", {})
    messages      = state.get("messages", [])
    context_msgs  = messages[-4:] if messages else []
    log_entries   = list(state.get("agent_log", []))

    data_summary = _get_data_summary(working_files)
    existing_insights = state.get("insights", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior data analyst. Based on the data summary below, generate
5-10 key business insights and findings.

FORMAT:
- Use bullet points
- Each insight should be specific and data-driven (reference actual numbers)
- Categorize as: 📊 Data Quality | 📈 Trends | 👥 Segments | ⚠️ Anomalies | 💡 Opportunities
- Include a "Next Steps" section at the bottom

Data Summary:
{summary}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    response = (prompt | coder_llm).invoke({
        "summary": data_summary[:4000],
        "messages": context_msgs,
    })

    insights_text = response.content
    # Parse insights into list
    new_insights = [
        line.strip().lstrip("•-* ")
        for line in insights_text.split("\n")
        if line.strip() and len(line.strip()) > 20
    ]

    log_entries.append(make_log_entry(
        "Chat", "Insights generated",
        f"Generated {len(new_insights)} insights", "success",
    ))
    return {
        "messages":        [AIMessage(content=insights_text)],
        "insights":        existing_insights + new_insights[:10],
        "analysis_result": insights_text,
        "error":           None,
        "agent_log":       log_entries,
        "next_step":       "done",
    }


def custom_code_node(state: MasterState) -> dict:
    """Run any custom Python analysis code the user requests."""
    working_files  = state.get("working_files", {})
    user_id        = state.get("user_id", "default")
    schema_info, _ = _get_schema(working_files)
    messages       = state.get("messages", [])
    context_msgs   = messages[-6:] if messages else []

    charts_dir = os.path.join("storage", user_id, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    error_feedback = (
        f"\n\nFIX THIS ERROR:\n{state.get('error')}" if state.get("error") else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python data scientist. Write a complete, self-contained analysis script
for exactly what the user asks for.

RULES:
1. Import all needed libraries (pandas, numpy, sklearn, matplotlib, scipy, statsmodels, etc.)
2. working_files dict is pre-defined: load data with pd.read_pickle(working_files[name])
3. charts_dir and user_id are pre-defined
4. For charts: import uuid; save to f"{{charts_dir}}/chart_{{uuid.uuid4().hex[:8]}}.png";
   print SAVED_CHART:<path>
5. Print all key results clearly
6. Handle errors gracefully with try/except

Schema: {schema}
Working files: {files}
Charts dir: {charts_dir}
{error_feedback}"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    response = (prompt | coder_llm).invoke({
        "files": json.dumps(working_files, indent=2),
        "schema": schema_info,
        "charts_dir": charts_dir,
        "error_feedback": error_feedback,
        "messages": context_msgs,
    })

    preamble = (
        f"import os\n"
        f"working_files = {json.dumps(working_files)}\n"
        f"charts_dir = '{charts_dir}'\n"
        f"user_id = '{user_id}'\n"
        f"os.makedirs(charts_dir, exist_ok=True)\n"
    )
    code   = _strip_code(response.content)
    result = repl_sandbox.run(preamble + code)

    log_entries      = list(state.get("agent_log", []))
    charts_generated = list(state.get("charts_generated", []))

    if result.get("error"):
        current_iter = state.get("iteration_count", 0) + 1
        if current_iter > 3:
            return {
                "messages":        [AIMessage(content=f"Analysis failed: {result['error'][:300]}")],
                "error":           result["error"],
                "iteration_count": current_iter,
                "next_step":       "done",
                "agent_log":       log_entries,
            }
        return {"error": result["error"], "iteration_count": current_iter,
                "next_step": "custom_code"}

    output_text = result.get("output", "<no output>")
    for m in re.finditer(r"SAVED_CHART:(.+\.png)", output_text):
        charts_generated.append(m.group(1).strip())

    # Summarize output
    if len(output_text) > 50:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Interpret this Python analysis output in clear, business-friendly language.
Include key findings and recommendations. Use markdown.

Output:
{output}"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        summary = coder_llm.invoke(
            summary_prompt.format_messages(output=output_text[:3000], messages=context_msgs)
        )
        final_content = summary.content
    else:
        final_content = output_text or "Analysis complete (no output)."

    log_entries.append(make_log_entry(
        "Chat", "Custom analysis complete", output_text[:100], "success",
    ))
    return {
        "messages":        [AIMessage(content=final_content)],
        "charts_generated": charts_generated,
        "analysis_result": final_content,
        "error":           None,
        "iteration_count": 0,
        "agent_log":       log_entries,
        "next_step":       "done",
    }


# ── Routing ───────────────────────────────────────────────────────────

def route_intent(state: MasterState) -> str:
    intent = state.get("next_step", "query")
    mapping = {
        "visualize":   "visualizer",
        "analytics":   "analytics",
        "statistics":  "statistics",
        "insights":    "insights",
        "custom_code": "custom_code",
    }
    return mapping.get(intent, "sql_query")


def route_retry(state: MasterState) -> str:
    """After any node: retry on error, or end."""
    if state.get("error"):
        if state.get("iteration_count", 0) > 3:
            return END
        next_step = state.get("next_step", "done")
        routing = {
            "query":       "sql_query",
            "visualize":   "visualizer",
            "analytics":   "analytics",
            "statistics":  "statistics",
            "custom_code": "custom_code",
        }
        return routing.get(next_step, END)
    return END


def build_chat_graph():
    workflow = StateGraph(MasterState)
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("sql_query",     sql_query_node)
    workflow.add_node("visualizer",    visualizer_node)
    workflow.add_node("analytics",     analytics_node)
    workflow.add_node("statistics",    statistics_node)
    workflow.add_node("insights",      insights_node)
    workflow.add_node("custom_code",   custom_code_node)

    workflow.set_entry_point("intent_router")
    workflow.add_conditional_edges("intent_router", route_intent, {
        "sql_query":   "sql_query",
        "visualizer":  "visualizer",
        "analytics":   "analytics",
        "statistics":  "statistics",
        "insights":    "insights",
        "custom_code": "custom_code",
    })

    # All analysis nodes can retry on error
    for node in ["sql_query", "visualizer", "analytics", "statistics", "custom_code"]:
        workflow.add_conditional_edges(node, route_retry, {
            "sql_query":   "sql_query",
            "visualizer":  "visualizer",
            "analytics":   "analytics",
            "statistics":  "statistics",
            "custom_code": "custom_code",
            END: END,
        })

    # Insights and statistics have no retry (they don't produce errors usually)
    workflow.add_edge("insights",   END)

    return workflow
