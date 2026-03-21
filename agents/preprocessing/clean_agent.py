"""
preprocessing/clean_agent.py

ReAct-style data cleaning agent.
The LLM reasons over the data using atomic inspection tools,
then writes free-form Python to fix whatever it finds.
No hardcoded problem list — new tools = new capabilities.
"""

import os, json, re, ast, shutil, textwrap, traceback, hashlib
import pandas as pd
import numpy as np
from typing import Any, Optional
from scipy import stats as scipy_stats
from pydantic import BaseModel, Field
from core.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.runnables.config import RunnableConfig
from core.state import MasterState
from core.sandbox import DockerREPL
from core.activity_log import make_log_entry
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# LLM + Sandbox
# ─────────────────────────────────────────────
llm = get_llm("coder", temperature=0.0)
repl_sandbox = DockerREPL()

AUTO_APPROVE = os.getenv("AUTO_APPROVE", "").lower() in ("1", "true", "yes")

MAX_TOOL_TURNS = 40          # hard stop on runaway loops
MAX_RESULT_CHARS = 6_000     # truncate large tool results before feeding back to LLM


# ─────────────────────────────────────────────
# NaN-safe JSON encoder
# ─────────────────────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        try:
            if pd.isna(obj): return None
        except Exception:
            pass
        return super().default(obj)


def _j(obj, indent=2) -> str:
    return json.dumps(obj, cls=SafeEncoder, indent=indent, ensure_ascii=False)


def _trunc(s: str, n: int = MAX_RESULT_CHARS) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated — {len(s)-n} chars omitted]"


# ─────────────────────────────────────────────
# Code-safety gate
# ─────────────────────────────────────────────
_FORBIDDEN = [
    "os.system", "subprocess", "eval(", "__import__",
    "open(", "exec(", "shutil.rmtree", "socket",
    "requests.", "urllib", "pickle.loads",
]

def _is_safe(code: str) -> tuple[bool, str]:
    for pat in _FORBIDDEN:
        if pat in code:
            return False, f"Forbidden pattern detected: '{pat}'"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    return True, ""


# ═══════════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
#  Each returns a plain dict — the dispatcher serialises to JSON
# ═══════════════════════════════════════════════════════════════════

def _load(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


# ── 1. OVERVIEW ─────────────────────────────────────────────────────
def tool_profile_dataframe(working_files: dict) -> dict:
    """
    High-level overview of every file: shape, dtypes, null counts,
    duplicate rows, memory, and a list of columns that may need attention.
    Call this FIRST before any other tool.
    """
    results = {}
    for name, path in working_files.items():
        try:
            df = _load(path)
            flagged = []
            col_summary = {}
            for col in df.columns:
                s = df[col]
                null_pct = s.isnull().mean() * 100
                dtype    = str(s.dtype)
                nunique  = s.nunique(dropna=True)
                reasons  = []

                if null_pct > 0:
                    reasons.append(f"{null_pct:.1f}% missing")
                if dtype == "object":
                    sample = s.dropna().astype(str)
                    # detect fake nulls
                    fake = sample.str.strip().isin(
                        ["?","N/A","n/a","NA","None","null","nan","NaN",""," ","-999","-9999","9999","#N/A","#NA","#NULL!","(blank)","missing","unknown","undefined","nil","NULL"]
                    ).sum()
                    if fake:
                        reasons.append(f"{fake} fake-null placeholders")
                    # detect mixed types
                    as_num = pd.to_numeric(sample, errors="coerce")
                    num_frac = as_num.notna().mean()
                    if 0.1 < num_frac < 0.9:
                        reasons.append("mixed numeric/text")
                    # detect possible date
                    if any(kw in col.lower() for kw in ["date","time","dt","timestamp","created","updated","at","on"]):
                        reasons.append("possible datetime column")
                    # high cardinality
                    if nunique > 0 and nunique / max(len(df),1) > 0.95:
                        reasons.append("very high cardinality — may be ID")
                if dtype in ("float64","int64","float32","int32"):
                    if s.dropna().std() == 0:
                        reasons.append("zero variance (constant column)")
                    if null_pct > 0:
                        reasons.append("numeric with nulls")
                if reasons:
                    flagged.append(col)
                col_summary[col] = {"dtype": dtype, "null_pct": round(null_pct,2), "nunique": int(nunique), "flags": reasons}

            results[name] = {
                "rows": len(df),
                "cols": len(df.columns),
                "duplicate_rows": int(df.duplicated().sum()),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
                "flagged_columns": flagged,
                "column_summary": col_summary,
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


# ── 2. DEEP COLUMN INSPECTION ────────────────────────────────────────
def tool_inspect_column(working_files: dict, filename: str, col: str) -> dict:
    """
    Full forensic report on a single column: distribution statistics,
    value counts, sample values, outlier bounds, data-type breakdown,
    detected patterns (email, phone, URL, JSON, currency, mixed units, etc.).
    Use this after profile_dataframe flags a column.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found. Available: {list(working_files.keys())}"}
    df = _load(path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not in {list(df.columns)}"}

    s = df[col]
    clean = s.dropna()
    total, n_null = len(s), int(s.isnull().sum())
    report: dict[str, Any] = {
        "dtype": str(s.dtype),
        "total_rows": total,
        "null_count": n_null,
        "null_pct": round(n_null / total * 100, 2) if total else 0,
        "nunique": int(clean.nunique()),
    }

    if len(clean) == 0:
        report["note"] = "Column is entirely null"
        return report

    # ── numeric checks
    numeric = pd.to_numeric(clean, errors="coerce")
    num_valid = numeric.dropna()
    if len(num_valid) / len(clean) > 0.6:
        q1, q3 = float(num_valid.quantile(0.25)), float(num_valid.quantile(0.75))
        iqr = q3 - q1
        report.update({
            "logical_type": "numeric",
            "mean": round(float(num_valid.mean()), 4),
            "median": round(float(num_valid.median()), 4),
            "std": round(float(num_valid.std()), 4),
            "min": float(num_valid.min()),
            "max": float(num_valid.max()),
            "q1": q1, "q3": q3,
            "skewness": round(float(num_valid.skew()), 3),
            "kurtosis": round(float(num_valid.kurt()), 3),
            "outliers_iqr_count": int(((num_valid < q1 - 1.5*iqr) | (num_valid > q3 + 1.5*iqr)).sum()),
            "negative_count": int((num_valid < 0).sum()),
            "zero_count": int((num_valid == 0).sum()),
            "non_numeric_rows": int(len(clean) - len(num_valid)),
        })
        return report

    # ── string checks
    str_s = clean.astype(str).str.strip()
    patterns: list[str] = []

    # fake nulls
    fake_null_vals = {"?","N/A","n/a","NA","None","null","nan","NaN","","-999","-9999","9999","#N/A","#NA","(blank)","missing","unknown","undefined","nil","NULL"}
    fake_count = str_s.isin(fake_null_vals).sum()
    if fake_count: patterns.append(f"fake_nulls:{fake_count}")

    # email
    if str_s.str.match(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$").mean() > 0.5:
        patterns.append("email")
    # phone
    if str_s.str.match(r"^\+?[\d\s\-().]{7,20}$").mean() > 0.5:
        patterns.append("phone_number")
    # URL
    if str_s.str.match(r"^https?://").mean() > 0.3:
        patterns.append("url")
    # JSON-like
    if str_s.str.startswith("{").mean() > 0.3 or str_s.str.startswith("[").mean() > 0.3:
        patterns.append("json_like")
    # currency / mixed units
    if str_s.str.match(r"^[$€£¥₹]?\s*[\d,.]+\s*[kKmMbB%]?$").mean() > 0.5:
        patterns.append("currency_or_mixed_units")
    # pure numeric string
    if str_s.str.match(r"^-?\d+\.?\d*$").mean() > 0.7:
        patterns.append("numeric_string — should be cast")
    # date-like
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}-[A-Za-z]{3}-\d{4}",
        r"[A-Za-z]+ \d{1,2},? \d{4}",
    ]
    for dp in date_patterns:
        if str_s.str.match(dp).mean() > 0.4:
            patterns.append("date_string — needs parsing")
            break
    # whitespace issues
    if (clean.astype(str) != clean.astype(str).str.strip()).any():
        patterns.append("leading_or_trailing_whitespace")
    # case inconsistency
    nuniq_lower = str_s.str.lower().nunique()
    if clean.nunique() > nuniq_lower:
        patterns.append(f"case_inconsistency: {clean.nunique()} raw vs {nuniq_lower} lowercased")

    # cardinality
    card_ratio = clean.nunique() / total
    if card_ratio > 0.95:
        report["logical_type"] = "id_or_high_cardinality_text"
    elif clean.nunique() <= 50 or card_ratio < 0.2:
        report["logical_type"] = "categorical"
    else:
        report["logical_type"] = "free_text"

    report["detected_patterns"] = patterns
    report["top_values"] = str_s.value_counts().head(20).to_dict()
    report["sample_values"] = clean.sample(min(10, len(clean)), random_state=42).tolist()
    report["whitespace_issues"] = int((clean.astype(str) != clean.astype(str).str.strip()).sum())
    return report


# ── 3. CORRELATION & REDUNDANCY ─────────────────────────────────────
def tool_check_correlations(working_files: dict, filename: str, threshold: float = 0.95) -> dict:
    """
    Find highly correlated numeric column pairs (potential redundancy).
    Also finds near-identical object columns. threshold default 0.95.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    num = df.select_dtypes(include=np.number).dropna(axis=1, how="all")
    pairs = []
    if len(num.columns) >= 2:
        corr = num.corr().abs()
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                c = corr.iloc[i, j]
                if not np.isnan(c) and c >= threshold:
                    pairs.append({"col_a": corr.columns[i], "col_b": corr.columns[j], "correlation": round(float(c), 4)})

    # near-duplicate object columns
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    dup_text = []
    for i in range(len(obj_cols)):
        for j in range(i+1, len(obj_cols)):
            try:
                same = (df[obj_cols[i]].astype(str) == df[obj_cols[j]].astype(str)).mean()
                if same >= threshold:
                    dup_text.append({"col_a": obj_cols[i], "col_b": obj_cols[j], "pct_identical": round(float(same)*100,2)})
            except Exception:
                pass
    return {"highly_correlated_numeric_pairs": pairs, "near_duplicate_text_columns": dup_text}


# ── 4. SCHEMA ANOMALIES ─────────────────────────────────────────────
def tool_detect_schema_anomalies(working_files: dict, filename: str) -> dict:
    """
    Detect structural problems: mixed types within a column, inconsistent
    row lengths for delimited fields, columns with only one unique value,
    columns that are entirely null, potential primary-key violations.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    issues = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            issues[col] = ["entirely_null"]
            continue
        col_issues = []
        # constant column
        if s.nunique() == 1:
            col_issues.append(f"constant_value: all rows = '{s.iloc[0]}'")
        # mixed types in object column
        if df[col].dtype == object:
            types = s.apply(type).value_counts()
            if len(types) > 1:
                col_issues.append(f"mixed_python_types: {types.to_dict()}")
            # delimiter inconsistency
            if s.astype(str).str.contains(",").mean() > 0.3:
                lengths = s.astype(str).str.split(",").apply(len)
                if lengths.std() > 1:
                    col_issues.append(f"delimited_field_variable_length: min={lengths.min()} max={lengths.max()}")
        # negative where it shouldn't be
        if any(kw in col.lower() for kw in ["age","price","cost","amount","salary","count","quantity","weight","height","distance","duration","size"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                neg = (df[col] < 0).sum()
                if neg:
                    col_issues.append(f"negative_values_in_likely_positive_col: {neg} rows")
        if col_issues:
            issues[col] = col_issues

    # primary key candidate check
    pk_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["id","key","uuid","code"])]
    pk_issues = {}
    for c in pk_candidates:
        dupes = int(df[c].duplicated(keep=False).sum())
        if dupes:
            pk_issues[c] = f"{dupes} duplicate values — not a clean primary key"
    return {"column_anomalies": issues, "pk_violations": pk_issues}


# ── 5. DISTRIBUTION ANALYSIS ────────────────────────────────────────
def tool_analyze_distributions(working_files: dict, filename: str) -> dict:
    """
    For every numeric column: skewness, kurtosis, outlier counts (IQR + z-score),
    zero-inflation, recommendation (log-transform, clip, standardize, etc.).
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    result = {}
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        skew = float(s.skew())
        kurt = float(s.kurt())
        iqr_out = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        z_out   = int((((s - s.mean()) / s.std()).abs() > 3).sum()) if s.std() > 0 else 0
        neg_count = int((s < 0).sum())
        zero_count = int((s == 0).sum())

        recs = []
        if abs(skew) > 1.5 and neg_count == 0:
            recs.append("log1p_transform — high skewness, all-positive values")
        if abs(skew) > 1.5 and neg_count > 0:
            recs.append("yeo_johnson_transform — high skewness with negatives")
        if iqr_out > len(s) * 0.01:
            recs.append(f"clip_or_cap_outliers — {iqr_out} IQR outliers ({round(iqr_out/len(s)*100,1)}%)")
        if z_out > len(s) * 0.005:
            recs.append(f"review_z_outliers — {z_out} rows beyond 3σ")
        if zero_count / len(s) > 0.4:
            recs.append(f"zero_inflated — {zero_count} zeros ({round(zero_count/len(s)*100,1)}%)")

        result[col] = {
            "skewness": round(skew,3), "kurtosis": round(kurt,3),
            "iqr_outliers": iqr_out, "z_outliers": z_out,
            "neg_count": neg_count, "zero_count": zero_count,
            "recommendations": recs,
        }
    return result


# ── 6. MISSING VALUE ANALYSIS ───────────────────────────────────────
def tool_analyze_missing(working_files: dict, filename: str) -> dict:
    """
    Detailed missing-value report: counts, percentages, MCAR/MAR heuristic,
    recommended imputation strategy per column.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    result = {}
    for col in df.columns:
        n_null = int(df[col].isnull().sum())
        if n_null == 0:
            continue
        pct = round(n_null / len(df) * 100, 2)
        dtype = str(df[col].dtype)
        rec = []

        if pct > 60:
            rec.append("consider_dropping_column — more than 60% missing")
        elif dtype in ("float64","int64","float32","int32"):
            skew = abs(float(df[col].skew())) if df[col].notna().sum() > 5 else 0
            if skew > 1:
                rec.append("fill_median — numeric, skewed distribution")
            else:
                rec.append("fill_mean_or_median")
            # check for groupby opportunity
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            for gc in cat_cols:
                if df[gc].nunique() <= 20:
                    rec.append(f"groupby_fill_median(groupby='{gc}') — low-cardinality category available")
                    break
        elif dtype == "object":
            top_mode_pct = (df[col].value_counts(normalize=True).iloc[0] * 100) if df[col].notna().any() else 0
            if top_mode_pct > 70:
                rec.append(f"fill_mode — dominant value covers {round(top_mode_pct,1)}% of non-nulls")
            else:
                rec.append("fill_constant('Unknown') — categorical with no dominant value")
        elif "datetime" in dtype:
            rec.append("forward_fill or backward_fill — datetime column")

        result[col] = {"null_count": n_null, "null_pct": pct, "dtype": dtype, "recommendations": rec}
    return result


# ── 7. DUPLICATE ANALYSIS ───────────────────────────────────────────
def tool_analyze_duplicates(working_files: dict, filename: str) -> dict:
    """
    Full duplicate analysis: exact row duplicates, near-duplicate
    detection on key string columns, potential duplicate-by-id.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    exact_dupes = int(df.duplicated().sum())
    subset_dupes = {}

    # find ID-like columns and check
    id_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["id","key","uuid","email"])]
    for c in id_cols:
        d = int(df[c].dropna().duplicated().sum())
        if d:
            subset_dupes[c] = f"{d} duplicate values in potential key column"

    # time-based duplicates if timestamp present
    dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    dt_info = {}
    for c in dt_cols:
        d = int(df[c].dropna().duplicated().sum())
        if d:
            dt_info[c] = f"{d} duplicate timestamps"

    return {
        "exact_duplicate_rows": exact_dupes,
        "potential_key_duplicates": subset_dupes,
        "duplicate_timestamps": dt_info,
        "recommendation": "use drop_duplicates() for exact; decide key policy for ID dupes",
    }


# ── 8. CATEGORICAL ANALYSIS ─────────────────────────────────────────
def tool_analyze_categories(working_files: dict, filename: str, col: str) -> dict:
    """
    Deep analysis of one categorical column: value counts, case variants,
    whitespace variants, rare categories, encoding recommendation.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    s = df[col].dropna().astype(str)
    nuniq = s.nunique()
    card_ratio = nuniq / max(len(df), 1)

    vc = s.value_counts()
    lower_vc = s.str.lower().str.strip().value_counts()
    case_variants = {}
    for val in lower_vc.index:
        raw = s[s.str.lower().str.strip() == val].unique().tolist()
        if len(raw) > 1:
            case_variants[val] = raw

    rare = vc[vc / len(df) < 0.01].index.tolist()
    rec = []
    if case_variants:
        rec.append("standardize_case_and_strip — multiple case variants detected")
    if rare:
        rec.append(f"group_{len(rare)}_rare_categories_as_Other — categories appear <1% of rows")
    if card_ratio > 0.5:
        rec.append("high_cardinality — consider frequency_encoding or target_encoding instead of one-hot")
    elif nuniq <= 10:
        rec.append("low_cardinality — safe for one-hot or label encoding")
    else:
        rec.append("moderate_cardinality — consider ordinal or frequency encoding")

    return {
        "nunique": nuniq,
        "cardinality_ratio": round(card_ratio, 4),
        "top_20_values": vc.head(20).to_dict(),
        "case_or_whitespace_variants": case_variants,
        "rare_categories_lt_1pct": rare[:30],
        "recommendations": rec,
    }


# ── 9. DATE/TIME ANALYSIS ───────────────────────────────────────────
def tool_analyze_datetimes(working_files: dict, filename: str) -> dict:
    """
    Detect date columns (parsed and unparsed), check for gaps, 
    time-zone issues, future dates, implausible dates, 
    and recommend feature engineering.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    result = {}
    candidates = list(df.select_dtypes(include=["datetime64"]).columns)
    # also check object columns with date-like names
    for c in df.columns:
        if c in candidates:
            continue
        if any(kw in c.lower() for kw in ["date","time","dt","timestamp","created","updated","birth","expir","start","end"]):
            sample = df[c].dropna().astype(str).head(20)
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.5:
                candidates.append(c)

    for col in candidates:
        s_raw = df[col]
        is_already_dt = pd.api.types.is_datetime64_any_dtype(s_raw)
        if is_already_dt:
            s = s_raw.dropna()
        else:
            s = pd.to_datetime(s_raw, errors="coerce", format="mixed").dropna()

        if len(s) == 0:
            result[col] = {"error": "could not parse as datetime"}
            continue

        now = pd.Timestamp.now()
        future = int((s > now).sum())
        ancient = int((s < pd.Timestamp("1900-01-01")).sum())

        info: dict[str, Any] = {
            "already_parsed": is_already_dt,
            "parse_success_pct": round(len(s) / len(df) * 100, 2),
            "min": str(s.min()), "max": str(s.max()),
            "future_dates": future,
            "pre_1900_dates": ancient,
            "timezone_aware": bool(s.dt.tz is not None) if is_already_dt else False,
            "recommendations": [],
        }
        if not is_already_dt:
            info["recommendations"].append("parse_to_datetime — column is string, needs pd.to_datetime")
        if future:
            info["recommendations"].append(f"review_{future}_future_dates — may be data entry errors")
        if ancient:
            info["recommendations"].append(f"review_{ancient}_pre_1900_dates — likely errors")
        info["recommendations"].append("engineer_features: year, month, day_of_week, is_weekend, days_since_reference")
        result[col] = info
    return result


# ── 10. TEXT QUALITY ANALYSIS ────────────────────────────────────────
def tool_analyze_text_quality(working_files: dict, filename: str, col: str) -> dict:
    """
    For free-text or semi-structured string columns: detect HTML/XML tags,
    special characters, encoding issues, excessive length variance,
    boilerplate/placeholder text, embedded JSON, multiline content.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    s = df[col].dropna().astype(str)
    lengths = s.str.len()
    issues = []
    if s.str.contains(r"<[^>]+>", regex=True).mean() > 0.1:
        issues.append("html_tags_present — strip with BeautifulSoup or regex")
    if s.str.contains(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", regex=True).mean() > 0.1:
        issues.append("unicode_escape_sequences — needs decoding")
    if s.str.contains(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", regex=True).any():
        issues.append("control_characters_present — strip non-printable chars")
    boilerplate = ["N/A","TBD","TBC","n/a","none","null","placeholder","test","lorem ipsum","todo","fill in","example"]
    bp_count = s.str.lower().str.strip().isin(boilerplate).sum()
    if bp_count:
        issues.append(f"boilerplate_values: {bp_count} rows with placeholder text")
    if (s.str.startswith("{") | s.str.startswith("[")).mean() > 0.2:
        issues.append("embedded_json — consider json_normalize to expand fields")
    if lengths.std() > lengths.mean() * 2:
        issues.append(f"high_length_variance — min={lengths.min()} max={lengths.max()} mean={round(float(lengths.mean()),1)}")
    if s.str.contains(r"\n|\r", regex=True).mean() > 0.1:
        issues.append("multiline_content — may need normalization")
    return {
        "row_count": len(s),
        "avg_length": round(float(lengths.mean()), 1),
        "max_length": int(lengths.max()),
        "min_length": int(lengths.min()),
        "detected_issues": issues,
        "sample_values": s.sample(min(5, len(s)), random_state=42).tolist(),
    }


# ── 11. MULTIVARIATE INCONSISTENCY ──────────────────────────────────
def tool_check_cross_column_consistency(working_files: dict, filename: str) -> dict:
    """
    Find logical inconsistencies between columns:
    end < start dates, negative profits despite positive revenue,
    age/birthdate mismatch, zip/state mismatch heuristics, etc.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    issues = []

    # start/end date pairs
    start_cols = [c for c in df.columns if any(k in c.lower() for k in ["start","begin","from","open","created","birth"])]
    end_cols   = [c for c in df.columns if any(k in c.lower() for k in ["end","finish","to","close","death","expir"])]
    for sc in start_cols:
        for ec in end_cols:
            try:
                s = pd.to_datetime(df[sc], errors="coerce")
                e = pd.to_datetime(df[ec], errors="coerce")
                bad = int((e < s).sum())
                if bad:
                    issues.append({"type":"date_order_violation", "col_start": sc, "col_end": ec, "count": bad})
            except Exception:
                pass

    # revenue / profit / cost triangles
    rev_cols  = [c for c in df.columns if any(k in c.lower() for k in ["revenue","sales","income","gross"])]
    cost_cols = [c for c in df.columns if any(k in c.lower() for k in ["cost","expense","cogs"])]
    prof_cols = [c for c in df.columns if any(k in c.lower() for k in ["profit","margin","net"])]
    for r in rev_cols:
        for c in cost_cols:
            for p in prof_cols:
                try:
                    implied = df[r] - df[c]
                    delta = (implied - df[p]).abs()
                    bad = int((delta > delta.quantile(0.99)).sum())
                    if bad > len(df) * 0.01:
                        issues.append({"type":"financial_inconsistency","revenue":r,"cost":c,"profit":p,"outlier_rows":bad})
                except Exception:
                    pass

    # age / birthdate consistency
    age_cols  = [c for c in df.columns if "age" in c.lower()]
    dob_cols  = [c for c in df.columns if any(k in c.lower() for k in ["birth","dob","born"])]
    for ac in age_cols:
        for dc in dob_cols:
            try:
                dob = pd.to_datetime(df[dc], errors="coerce")
                now = pd.Timestamp.now()
                implied_age = (now - dob).dt.days / 365.25
                bad = int(((df[ac] - implied_age).abs() > 2).sum())
                if bad:
                    issues.append({"type":"age_birthdate_mismatch","age_col":ac,"dob_col":dc,"mismatch_rows":bad})
            except Exception:
                pass

    return {"cross_column_issues": issues, "total_issues_found": len(issues)}


# ── 12. ENCODING ISSUES ──────────────────────────────────────────────
def tool_detect_encoding_issues(working_files: dict, filename: str) -> dict:
    """
    Scan string columns for mojibake (encoding corruption), 
    non-ASCII characters, invisible characters, RTL markers,
    and null bytes embedded in strings.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    result = {}
    for col in df.select_dtypes(include="object").columns:
        s = df[col].dropna().astype(str)
        issues = []
        # mojibake heuristic: common replacement chars
        if s.str.contains("â€|Ã|â‚¬|Â", regex=True, na=False).any():
            issues.append("likely_mojibake — UTF-8 decoded as latin-1")
        # non-ASCII
        non_ascii = int(s.apply(lambda x: any(ord(c) > 127 for c in x)).sum())
        if non_ascii:
            issues.append(f"non_ascii_chars_in_{non_ascii}_rows")
        # null bytes
        if s.str.contains("\x00", na=False).any():
            issues.append("null_bytes_present — strip \\x00")
        # RTL / BOM markers
        if s.str.contains("[\u200e\u200f\ufeff\u202a-\u202e]", regex=True, na=False).any():
            issues.append("unicode_directionality_or_BOM_markers")
        # zero-width spaces
        if s.str.contains("[\u200b\u200c\u200d\u2060]", regex=True, na=False).any():
            issues.append("zero_width_characters — invisible but affect comparisons")
        if issues:
            result[col] = issues
    return result


# ── 13. OUTLIER DEEP DIVE ────────────────────────────────────────────
def tool_inspect_outliers(working_files: dict, filename: str, col: str) -> dict:
    """
    Detailed outlier inspection for one numeric column:
    IQR bounds, z-score bounds, extreme values, recommendation
    (clip, log-transform, investigate, flag-as-boolean).
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 4:
        return {"error": "Not enough numeric data"}

    q1, q3  = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr     = q3 - q1
    lo_iqr  = q1 - 1.5 * iqr
    hi_iqr  = q3 + 1.5 * iqr
    lo_ext  = q1 - 3.0 * iqr
    hi_ext  = q3 + 3.0 * iqr

    mild_out = s[(s < lo_iqr) | (s > hi_iqr)]
    ext_out  = s[(s < lo_ext) | (s > hi_ext)]

    mean, std = float(s.mean()), float(s.std())
    z_out = s[(s - mean).abs() / std > 3] if std > 0 else pd.Series([], dtype=float)

    rec = []
    skew = abs(float(s.skew()))
    if skew > 2 and s.min() >= 0:
        rec.append("log1p_transform — extreme positive skew; log transform may handle outliers naturally")
    elif len(ext_out) < len(s) * 0.005:
        rec.append("clip_at_extreme_IQR_bounds — very few extreme outliers, safe to cap")
    elif len(mild_out) < len(s) * 0.02:
        rec.append("clip_at_1pct_99pct_quantiles — small proportion of outliers")
    else:
        rec.append("investigate_before_clipping — >2% rows are outliers, may be legitimate data")
    if s.min() < 0 and any(kw in col.lower() for kw in ["age","price","amount","count","weight","height"]):
        rec.append("flag_negatives_as_error — negative values in a likely-positive column")

    return {
        "count": len(s),
        "mean": round(mean,4), "std": round(std,4),
        "min": float(s.min()), "max": float(s.max()),
        "q1": q1, "q3": q3, "iqr": round(iqr,4),
        "iqr_lower_fence": round(lo_iqr,4), "iqr_upper_fence": round(hi_iqr,4),
        "extreme_lower": round(lo_ext,4),   "extreme_upper": round(hi_ext,4),
        "mild_outlier_count": len(mild_out),
        "extreme_outlier_count": len(ext_out),
        "z_score_outlier_count": len(z_out),
        "extreme_sample_low":  ext_out[ext_out < lo_ext].head(5).tolist(),
        "extreme_sample_high": ext_out[ext_out > hi_ext].head(5).tolist(),
        "recommendations": rec,
    }


# ── 14. FEATURE ENGINEERING SUGGESTIONS ─────────────────────────────
def tool_suggest_feature_engineering(working_files: dict, filename: str) -> dict:
    """
    Scan the dataset and suggest valuable derived features:
    date decomposition, ratio features, interaction terms,
    binning, cyclical encoding for months/hours, text length features.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    suggestions = []

    # datetime decomposition
    dt_cols = list(df.select_dtypes(include="datetime64").columns)
    for c in dt_cols:
        suggestions.append({"column": c, "feature": f"{c}_year, {c}_month, {c}_dayofweek, {c}_is_weekend", "reason": "standard datetime decomposition"})
        s = df[c].dropna()
        if len(s) > 0:
            span_days = (s.max() - s.min()).days
            if span_days > 365:
                suggestions.append({"column": c, "feature": f"{c}_days_since_min", "reason": "useful ordinal time feature for ML"})

    # numeric ratios
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for i, a in enumerate(num_cols):
        for b in num_cols[i+1:]:
            if any(kw in a.lower() for kw in ["revenue","sales","income"]) and any(kw in b.lower() for kw in ["cost","expense"]):
                suggestions.append({"columns": [a,b], "feature": f"{a}_to_{b}_ratio", "reason": "financial efficiency ratio"})
            if any(kw in a.lower() for kw in ["profit","net"]) and any(kw in b.lower() for kw in ["revenue","sales"]):
                suggestions.append({"columns": [a,b], "feature": f"{a}_margin_pct", "reason": "profit margin percentage"})

    # text length
    for c in df.select_dtypes(include="object").columns:
        s = df[c].dropna().astype(str)
        if s.str.len().mean() > 20:
            suggestions.append({"column": c, "feature": f"{c}_char_length, {c}_word_count", "reason": "text length may correlate with target"})

    # high-cardinality → frequency encode
    for c in df.select_dtypes(include="object").columns:
        if 10 < df[c].nunique() < 500:
            suggestions.append({"column": c, "feature": f"{c}_frequency_encoded", "reason": f"moderate cardinality ({df[c].nunique()} unique) — frequency encode"})

    # cyclical encoding
    for c in df.columns:
        if "month" in c.lower():
            suggestions.append({"column": c, "feature": f"{c}_sin, {c}_cos", "reason": "cyclical encoding for month (1-12 wrap-around)"})
        if "hour" in c.lower():
            suggestions.append({"column": c, "feature": f"{c}_sin, {c}_cos", "reason": "cyclical encoding for hour (0-23 wrap-around)"})
        if "day" in c.lower() and "of" in c.lower():
            suggestions.append({"column": c, "feature": f"{c}_sin, {c}_cos", "reason": "cyclical encoding for day of week"})

    return {"suggestions": suggestions[:30]}  # cap at 30 to avoid token explosion


# ── 15. DATA LEAKAGE SCAN ────────────────────────────────────────────
def tool_scan_data_leakage(working_files: dict, filename: str, target_col: Optional[str] = None) -> dict:
    """
    Heuristic scan for potential data leakage: columns with suspiciously
    high correlation to target, future-derived columns (e.g. 'result',
    'outcome', 'label', 'target', 'score'), and ID columns that shouldn't
    be features.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    warnings = []

    # name-based leakage suspects
    leakage_keywords = ["target","label","outcome","result","prediction","score","flag","churn","converted","purchased","clicked","approved","defaulted","fraud","is_"]
    for c in df.columns:
        if c == target_col:
            continue
        if any(kw in c.lower() for kw in leakage_keywords):
            warnings.append({"column": c, "reason": "column name suggests it may encode the target — verify it's not a post-event derived feature"})

    # ID columns as features
    for c in df.columns:
        if any(kw in c.lower() for kw in ["_id","id_","uuid","key"]) and c != target_col:
            warnings.append({"column": c, "reason": "ID column — should not be used as ML feature"})

    # high correlation to target
    if target_col and target_col in df.columns:
        num = df.select_dtypes(include=np.number)
        if target_col in num.columns:
            corr = num.corr()[target_col].abs().drop(target_col, errors="ignore")
            very_high = corr[corr > 0.98]
            for c, v in very_high.items():
                warnings.append({"column": c, "reason": f"correlation={round(float(v),4)} with target — possible leakage"})

    return {"leakage_warnings": warnings, "total": len(warnings)}


# ── 16. RUN TRANSFORMATION (the "hands") ─────────────────────────────
def tool_run_transformation(working_files: dict, code: str, description: str) -> dict:
    """
    Execute Python code to transform the data.
    The code receives `working_files` dict, loads each file with
    pd.read_pickle, transforms it, and saves with df.to_pickle(path).
    NEVER use dropna() unless deduplication. NEVER drop columns without reason.
    Return a success/error dict.
    """
    safe, reason = _is_safe(code)
    if not safe:
        return {"status": "blocked", "reason": reason}

    # backup
    backups = {}
    for name, path in working_files.items():
        if os.path.exists(path):
            bk = path + ".bak"
            shutil.copy(path, bk)
            backups[name] = bk

    # inject working_files into the code
    preamble = textwrap.dedent(f"""
import pandas as pd
import numpy as np
import re, json
working_files = {json.dumps(working_files)}
""")
    result = repl_sandbox.run(preamble + "\n" + code)

    if result.get("error"):
        # restore backups
        for name, bk in backups.items():
            if os.path.exists(bk):
                shutil.move(bk, working_files[name])
        return {"status": "error", "error": result["error"]}

    # clean up backups on success
    for bk in backups.values():
        if os.path.exists(bk):
            os.remove(bk)

    return {"status": "success", "stdout": result.get("stdout", "")[:2000]}


# ── 17. VERIFY RESULT ────────────────────────────────────────────────
def tool_verify_result(working_files: dict, filename: str, checks: Optional[list] = None) -> dict:
    """
    Post-transformation audit. Returns: row/col counts, null diffs,
    dtype changes, new constant columns, new entirely-null columns.
    Pass `checks` as list of column names to focus on, or omit for full scan.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    bk_path = path + ".bak"

    result: dict[str, Any] = {
        "rows": len(df),
        "cols": len(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(df[c].isnull().sum()) for c in df.columns},
        "new_constant_cols": [c for c in df.columns if df[c].nunique() <= 1],
        "entirely_null_cols": [c for c in df.columns if df[c].isnull().all()],
    }

    if os.path.exists(bk_path):
        try:
            old = pd.read_pickle(bk_path)
            result["row_delta"] = len(df) - len(old)
            result["col_delta"] = len(df.columns) - len(old.columns)
            null_diff = {}
            for c in df.columns:
                if c in old.columns:
                    diff = int(df[c].isnull().sum()) - int(old[c].isnull().sum())
                    if diff != 0:
                        null_diff[c] = diff
            result["null_count_delta"] = null_diff
            dtype_changes = {}
            for c in df.columns:
                if c in old.columns and str(df[c].dtype) != str(old[c].dtype):
                    dtype_changes[c] = {"before": str(old[c].dtype), "after": str(df[c].dtype)}
            result["dtype_changes"] = dtype_changes
        except Exception as e:
            result["backup_compare_error"] = str(e)

    return result


# ── 18. GET SAMPLE ROWS ──────────────────────────────────────────────
def tool_get_sample_rows(working_files: dict, filename: str, n: int = 10, condition: Optional[str] = None) -> dict:
    """
    Return n sample rows, optionally filtered by a pandas query string
    (e.g., condition="age < 0" or condition="status == 'UNKN'").
    Useful for sanity-checking after a transformation.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    try:
        subset = df.query(condition) if condition else df
        sample = subset.sample(min(n, len(subset)), random_state=42) if len(subset) > 0 else subset.head(n)
        return {"rows": json.loads(_j(sample.to_dict(orient="records")))}
    except Exception as e:
        return {"error": str(e)}


# ── 19. COLUMN STATISTICS DIFF ───────────────────────────────────────
def tool_compare_before_after(working_files: dict, filename: str, col: str) -> dict:
    """
    Compare current state of a column vs its backed-up version.
    Shows null count, nunique, dtype, mean/mode before and after.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    bk = path + ".bak"
    df_new = _load(path)

    if col not in df_new.columns:
        return {"error": f"Column '{col}' not in current data"}

    def stats(s):
        d = {"dtype": str(s.dtype), "null_count": int(s.isnull().sum()), "nunique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            d["mean"]   = round(float(s.mean()), 4) if s.notna().any() else None
            d["median"] = round(float(s.median()), 4) if s.notna().any() else None
            d["std"]    = round(float(s.std()), 4) if s.notna().any() else None
        else:
            mode = s.mode()
            d["mode"] = str(mode.iloc[0]) if len(mode) else None
        return d

    result = {"after": stats(df_new[col])}
    if os.path.exists(bk):
        try:
            df_old = pd.read_pickle(bk)
            if col in df_old.columns:
                result["before"] = stats(df_old[col])
        except Exception:
            pass
    return result


# ── 20. REQUEST HUMAN INPUT ──────────────────────────────────────────
def tool_request_human_input(question: str) -> dict:
    """
    Use ONLY when a domain decision cannot be made from data alone.
    Example: 'Should -1 in the priority column mean high priority or be treated as missing?'
    This will pause the agent and ask the user.
    """
    return {"__human_question__": question}


# ── 21. FUZZY DUPLICATE DETECTION  (Problem #2 partial) ─────────────
def tool_detect_fuzzy_duplicates(working_files: dict, filename: str, col: str, threshold: int = 90) -> dict:
    """
    Find near-duplicate string values within a column using edit-distance
    similarity (e.g. 'John Smith' vs 'John Simth', 'USA' vs 'US').
    threshold: 0-100 similarity score (default 90 = very similar).
    Returns candidate pairs. Useful for name/address/category columns.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    if col not in df.columns:
        return {"error": f"Column '{col}' not found"}

    try:
        from difflib import SequenceMatcher
    except ImportError:
        return {"error": "difflib not available"}

    vals = df[col].dropna().astype(str).str.strip().unique().tolist()
    if len(vals) > 2000:
        vals = vals[:2000]  # cap to avoid O(n^2) explosion

    pairs = []
    thresh = threshold / 100.0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            ratio = SequenceMatcher(None, vals[i].lower(), vals[j].lower()).ratio()
            if ratio >= thresh:
                pairs.append({
                    "value_a": vals[i],
                    "value_b": vals[j],
                    "similarity_pct": round(ratio * 100, 1),
                    "count_a": int((df[col].astype(str).str.strip() == vals[i]).sum()),
                    "count_b": int((df[col].astype(str).str.strip() == vals[j]).sum()),
                })
    pairs.sort(key=lambda x: -x["similarity_pct"])
    return {
        "col": col, "threshold_pct": threshold,
        "candidate_pairs": pairs[:40],
        "note": "Use run_transformation to apply a mapping dict if pairs are genuine duplicates",
    }


# ── 22. RANGE & BUSINESS RULE VALIDATION  (Problems #5, #8) ─────────
def tool_validate_ranges(working_files: dict, filename: str) -> dict:
    """
    Detect values that violate common domain constraints:
    - Negative values in inherently positive columns (age, price, count, weight...)
    - Ages > 120 or < 0
    - Percentages outside 0-100
    - Dates far in future or past
    - Columns where >5% of values are the same suspicious sentinel (-1, 0, 9999...)
    Returns per-column violations with row counts and sample bad values.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    violations = {}

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        col_issues = []

        # domain-name based rules
        clow = col.lower()
        if any(k in clow for k in ["age", "tenure", "years", "experience"]):
            bad = s[(s < 0) | (s > 120)]
            if len(bad):
                col_issues.append({"rule": "age_range_0_120", "violation_count": len(bad), "samples": bad.head(5).tolist()})

        if any(k in clow for k in ["price", "cost", "amount", "salary", "revenue", "income", "fee", "payment"]):
            bad = s[s < 0]
            if len(bad):
                col_issues.append({"rule": "non_negative_financial", "violation_count": len(bad), "samples": bad.head(5).tolist()})

        if any(k in clow for k in ["pct", "percent", "rate", "ratio", "score"]) and s.max() <= 1.1:
            bad = s[(s < 0) | (s > 1)]
            if len(bad):
                col_issues.append({"rule": "proportion_0_to_1", "violation_count": len(bad), "samples": bad.head(5).tolist()})
        elif any(k in clow for k in ["pct", "percent"]) and s.max() > 1:
            bad = s[(s < 0) | (s > 100)]
            if len(bad):
                col_issues.append({"rule": "percentage_0_to_100", "violation_count": len(bad), "samples": bad.head(5).tolist()})

        if any(k in clow for k in ["count", "quantity", "num_", "n_", "cnt"]):
            bad = s[s < 0]
            if len(bad):
                col_issues.append({"rule": "count_non_negative", "violation_count": len(bad), "samples": bad.head(5).tolist()})

        # sentinel value check: is -1 / 0 / 9999 suspiciously frequent?
        for sentinel in [-1, -999, -9999, 9999, 999, 0]:
            sentinel_pct = (s == sentinel).mean()
            if sentinel_pct > 0.05:
                col_issues.append({"rule": f"sentinel_value_{sentinel}_too_frequent", "pct": round(sentinel_pct * 100, 2)})

        if col_issues:
            violations[col] = col_issues

    return {"range_violations": violations, "total_columns_with_violations": len(violations)}


# ── 23. MULTICOLLINEARITY / VIF  (Problem #11) ──────────────────────
def tool_check_multicollinearity(working_files: dict, filename: str, vif_threshold: float = 5.0) -> dict:
    """
    Compute Variance Inflation Factor (VIF) for all numeric columns.
    VIF > 5 = moderate multicollinearity; VIF > 10 = severe.
    Also returns correlation matrix for columns with |corr| > 0.7.
    Use this to decide which redundant features to drop before ML.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    num = df.select_dtypes(include=np.number).dropna(axis=1, how="all")
    num_clean = num.dropna()

    if len(num_clean.columns) < 2:
        return {"note": "Fewer than 2 numeric columns — VIF not applicable"}
    if len(num_clean) < len(num_clean.columns) + 2:
        return {"note": "Not enough rows to compute VIF reliably"}

    try:
        from numpy.linalg import matrix_rank
        X = num_clean.values
        if matrix_rank(X) < X.shape[1]:
            return {"note": "Matrix is rank-deficient (perfect multicollinearity). Drop one of the identical columns first."}

        vif_results = {}
        for i, col in enumerate(num_clean.columns):
            others = np.delete(X, i, axis=1)
            r2 = 1 - (np.var(X[:, i] - others @ np.linalg.lstsq(others, X[:, i], rcond=None)[0])
                      / np.var(X[:, i])) if np.var(X[:, i]) > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else float("inf")
            vif_results[col] = round(float(vif), 2)

        high_vif = {c: v for c, v in vif_results.items() if v > vif_threshold}

        # high-correlation pairs
        corr = num_clean.corr().abs()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                c = corr.iloc[i, j]
                if not np.isnan(c) and c > 0.7:
                    high_corr.append({"col_a": corr.columns[i], "col_b": corr.columns[j], "abs_corr": round(float(c), 3)})

        return {
            "vif_scores": vif_results,
            "high_vif_columns": high_vif,
            "high_correlation_pairs_gt_07": sorted(high_corr, key=lambda x: -x["abs_corr"]),
            "recommendation": (
                "Drop or combine high-VIF columns before regression models. "
                "For tree-based models, multicollinearity is less critical."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


# ── 24. SCALING ANALYSIS  (Problem #12, #27) ────────────────────────
def tool_analyze_scaling(working_files: dict, filename: str) -> dict:
    """
    Check whether numeric columns need scaling/normalization.
    Reports: current range, mean, std, magnitude difference across columns,
    whether columns appear already scaled, and recommended scaler type.
    Also detects if training vs test split scaling was applied inconsistently
    by checking if any numeric column has been standardized (mean≈0, std≈1).
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    result = {}
    magnitudes = {}

    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        mn, mx = float(s.min()), float(s.max())
        mean, std = float(s.mean()), float(s.std())
        skew = float(s.skew())
        rng = mx - mn
        magnitudes[col] = rng

        rec = []
        already_standardized = abs(mean) < 0.1 and 0.9 < std < 1.1
        already_minmax = 0 <= mn and mx <= 1 and rng < 1.01

        if already_standardized:
            rec.append("appears_already_z_score_standardized")
        elif already_minmax:
            rec.append("appears_already_min_max_scaled")
        else:
            if abs(skew) > 1.5 and mn >= 0:
                rec.append("log1p_then_standardize — skewed positive column")
            elif s[(s - mean).abs() > 3 * std].mean() > 0.01:
                rec.append("robust_scaler — significant outliers present")
            else:
                rec.append("standard_scaler (z-score) — normal-ish distribution")

        result[col] = {
            "min": round(mn, 4), "max": round(mx, 4),
            "mean": round(mean, 4), "std": round(std, 4),
            "range": round(rng, 4), "skewness": round(skew, 3),
            "recommendation": rec,
        }

    # cross-column magnitude warning
    if magnitudes:
        max_mag = max(magnitudes.values())
        min_mag = min(v for v in magnitudes.values() if v > 0) if any(v > 0 for v in magnitudes.values()) else 1
        ratio = max_mag / min_mag if min_mag > 0 else 0
        result["__summary__"] = {
            "magnitude_ratio_max_to_min": round(ratio, 1),
            "scaling_needed": ratio > 100,
            "note": "Columns differ by >100x in range — distance-based models (KNN, SVM, gradient descent) need scaling." if ratio > 100 else "Ranges are reasonably similar.",
        }

    return result


# ── 25. STRUCTURAL ISSUES  (Problem #16) ────────────────────────────
def tool_detect_structural_issues(working_files: dict, filename: str) -> dict:
    """
    Detect structural data problems:
    - Column names with spaces, special chars, inconsistent casing
    - Multiple values packed into one column (comma/pipe/semicolon delimited)
    - Columns that look like they should be rows (wide → long pivot needed)
    - Numeric column names (unnamed columns from bad CSV parsing)
    - Columns that are entirely one data type but named as another
    - Repeated column name patterns suggesting one variable split across many cols
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    issues = []

    # column name issues
    bad_names = []
    for c in df.columns:
        cs = str(c)
        if " " in cs:
            bad_names.append({"col": cs, "issue": "contains_space"})
        if re.search(r"[^a-zA-Z0-9_]", cs):
            bad_names.append({"col": cs, "issue": f"special_chars: {re.findall(r'[^a-zA-Z0-9_]', cs)}"})
        if cs != cs.lower() and cs != cs.upper():
            bad_names.append({"col": cs, "issue": "mixed_case — prefer snake_case"})
        if re.match(r"^Unnamed", cs):
            bad_names.append({"col": cs, "issue": "unnamed_column — likely index column from CSV"})
        if re.match(r"^\d+$", cs):
            bad_names.append({"col": cs, "issue": "numeric_column_name — CSV parsing error"})
    if bad_names:
        issues.append({"type": "column_naming", "details": bad_names[:20]})

    # multi-value packed in one column
    for col in df.select_dtypes(include="object").columns:
        s = df[col].dropna().astype(str)
        for delim in [",", "|", ";"]:
            if s.str.contains(re.escape(delim), regex=False).mean() > 0.4:
                sample_parts = s.str.split(delim).apply(len)
                if sample_parts.std() < 2:
                    issues.append({
                        "type": "multi_value_packed",
                        "col": col,
                        "delimiter": delim,
                        "avg_values_per_cell": round(float(sample_parts.mean()), 1),
                        "fix": f"Use df['{col}'].str.split('{delim}', expand=True) or get_dummies(sep='{delim}')",
                    })
                break

    # repeated column name pattern suggesting wide-format (e.g. month_1, month_2, ...)
    col_prefixes: dict[str, list] = {}
    for c in df.columns:
        m = re.match(r"^([a-zA-Z_]+)_?(\d+)$", str(c))
        if m:
            col_prefixes.setdefault(m.group(1), []).append(c)
    for prefix, cols in col_prefixes.items():
        if len(cols) >= 3:
            issues.append({
                "type": "wide_format_detected",
                "prefix": prefix,
                "columns": cols,
                "fix": f"Consider pd.melt(df, value_vars={cols}, var_name='{prefix}_period', value_name='{prefix}_value')",
            })

    return {"structural_issues": issues, "total": len(issues)}


# ── 26. UNIT INCONSISTENCY DETECTION  (Problem #17) ─────────────────
def tool_detect_unit_inconsistencies(working_files: dict, filename: str) -> dict:
    """
    Detect columns that may contain mixed measurement units:
    - Numeric columns with bimodal distributions (e.g. some rows in kg, others in lbs)
    - String columns containing mixed unit suffixes (kg/lbs, km/miles, °C/°F)
    - Implausible value jumps (e.g. weight column switching between ~70 and ~154)
    Returns suspected columns with distribution evidence and conversion suggestions.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    findings = {}

    # string columns with explicit unit suffixes
    unit_patterns = {
        "weight": (r"\b(kg|lbs?|pounds?|kilograms?|grams?|oz)\b", "standardize to kg: lbs*0.4536, oz*0.02835"),
        "temperature": (r"\b(°?[CF]|celsius|fahrenheit|kelvin)\b", "standardize to Celsius: (°F-32)*5/9, K-273.15"),
        "distance": (r"\b(km|miles?|meters?|feet|ft|inches?|cm)\b", "standardize to km: miles*1.6093, m/1000"),
        "currency": (r"[$€£¥₹]", "standardize to single currency using exchange rate"),
    }
    for col in df.select_dtypes(include="object").columns:
        s = df[col].dropna().astype(str).str.lower()
        for domain, (pattern, fix) in unit_patterns.items():
            matches = s.str.extract(f"({pattern})", expand=False).dropna()
            if len(matches) > 10:
                unique_units = matches.str.lower().unique().tolist()
                if len(unique_units) > 1:
                    findings[col] = {
                        "domain": domain,
                        "mixed_units_found": unique_units[:10],
                        "affected_rows": len(matches),
                        "fix_suggestion": fix,
                    }
                break

    # numeric columns: bimodal detection via simple histogram gap
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if len(s) < 50:
            continue
        clow = col.lower()
        # weight: human weight in kg (40-150) vs lbs (88-330) — ratio ~2.2
        if any(k in clow for k in ["weight", "mass"]):
            light = ((s >= 30) & (s <= 160)).sum()
            heavy = ((s > 160) & (s <= 400)).sum()
            if light > 10 and heavy > 10 and min(light, heavy) / max(light, heavy) > 0.1:
                findings[col] = {
                    "domain": "weight",
                    "suspicion": f"Bimodal: {light} values in kg range (30-160), {heavy} in lbs range (160-400)",
                    "fix_suggestion": "Identify source of each row and convert lbs to kg (*0.4536)",
                }
        # temperature: Celsius (-50 to 60 typical) vs Fahrenheit (-58 to 140)
        if any(k in clow for k in ["temp", "temperature"]):
            celsius_range = ((s >= -50) & (s <= 60)).mean()
            if celsius_range < 0.5:
                findings[col] = {
                    "domain": "temperature",
                    "suspicion": f"Only {round(celsius_range*100,1)}% of values in typical Celsius range — may be Fahrenheit",
                    "fix_suggestion": "If Fahrenheit: df[col] = (df[col] - 32) * 5/9",
                }

    return {"unit_inconsistencies": findings, "total": len(findings)}


# ── 27. SPARSE DATA ANALYSIS  (Problem #19) ─────────────────────────
def tool_analyze_sparsity(working_files: dict, filename: str) -> dict:
    """
    Measure sparsity: zero-fill rate and null-fill rate per column and overall.
    Identifies columns that are mostly zero (sparse features) vs truly missing.
    Recommends sparse representation, feature selection, or dimensionality reduction.
    Also flags one-hot-encoded columns (all 0/1 binary) for potential compression.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    total_cells = df.shape[0] * df.shape[1]
    total_zeros = 0
    total_nulls = int(df.isnull().sum().sum())
    col_results = {}

    for col in df.columns:
        s = df[col]
        n_null = int(s.isnull().sum())
        n_zero = int((s == 0).sum()) if pd.api.types.is_numeric_dtype(s) else 0
        n_total = len(s)
        zero_pct = round(n_zero / n_total * 100, 1) if n_total else 0
        null_pct = round(n_null / n_total * 100, 1) if n_total else 0
        sparse_pct = round((n_zero + n_null) / n_total * 100, 1) if n_total else 0
        total_zeros += n_zero

        if sparse_pct > 70:
            # check if binary (one-hot candidate)
            uniq = s.dropna().unique()
            is_binary = set(uniq).issubset({0, 1, 0.0, 1.0})
            col_results[col] = {
                "zero_pct": zero_pct,
                "null_pct": null_pct,
                "sparse_pct": sparse_pct,
                "is_binary_01": is_binary,
                "recommendation": (
                    "one_hot_column — consider storing as sparse dtype or combining low-frequency flags"
                    if is_binary else
                    "sparse_numeric — consider scipy.sparse or feature_selection to reduce dimensionality"
                ),
            }

    overall_sparsity = round((total_zeros + total_nulls) / max(total_cells, 1) * 100, 1)
    return {
        "overall_sparsity_pct": overall_sparsity,
        "total_nulls": total_nulls,
        "total_zeros": total_zeros,
        "sparse_columns_gt70pct": col_results,
        "note": (
            "If overall sparsity >70%, consider PCA / TruncatedSVD before ML. "
            "Use pd.SparseDtype for memory efficiency."
        ) if overall_sparsity > 70 else "Sparsity within acceptable range.",
    }


# ── 28. CLASS IMBALANCE ANALYSIS  (Problem #9) ──────────────────────
def tool_analyze_class_imbalance(working_files: dict, filename: str, target_col: str) -> dict:
    """
    Analyse class distribution in a target/label column.
    Returns: class counts, imbalance ratio, minority class percentage,
    and recommended handling strategy (SMOTE, class weights, threshold tuning).
    Works for binary and multiclass targets.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found"}

    vc = df[target_col].value_counts()
    total = len(df[target_col].dropna())
    class_pcts = {str(k): round(v / total * 100, 2) for k, v in vc.items()}
    majority = int(vc.iloc[0])
    minority = int(vc.iloc[-1])
    imbalance_ratio = round(majority / minority, 1) if minority > 0 else float("inf")
    minority_pct = round(minority / total * 100, 2)

    recs = []
    if imbalance_ratio < 3:
        recs.append("balanced_enough — imbalance ratio < 3:1, standard training should work")
    elif imbalance_ratio < 10:
        recs.append("class_weight='balanced' in sklearn estimators — moderate imbalance")
        recs.append("adjust_decision_threshold — tune probability cutoff on validation set")
    else:
        recs.append(f"severe_imbalance ({imbalance_ratio}:1) — use SMOTE oversampling on training set only")
        recs.append("use_class_weight='balanced' AND oversample minority with SMOTE")
        recs.append("use_precision_recall_AUC not accuracy — accuracy is misleading here")
        recs.append("CRITICAL: apply SMOTE ONLY after train/test split, never on full dataset")

    return {
        "target_col": target_col,
        "class_distribution": class_pcts,
        "majority_count": majority,
        "minority_count": minority,
        "imbalance_ratio": f"{imbalance_ratio}:1",
        "minority_class_pct": minority_pct,
        "recommendations": recs,
    }


# ── 29. PII / SENSITIVE DATA SCAN  (Problem #30) ────────────────────
def tool_scan_pii(working_files: dict, filename: str) -> dict:
    """
    Detect columns likely containing Personally Identifiable Information (PII):
    names, emails, phone numbers, SSNs, credit cards, IP addresses,
    physical addresses, dates of birth.
    Returns column names, PII type, sample (masked), and anonymization options.
    Does NOT return actual PII values — samples are always masked.
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    findings = {}

    pii_patterns = {
        "email":       r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}",
        "phone":       r"\+?[\d\s\-().]{10,15}",
        "ssn_us":      r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "credit_card": r"\b(?:\d[ -]?){13,16}\b",
        "ip_address":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "zip_code_us": r"\b\d{5}(?:[-\s]\d{4})?\b",
        "url":         r"https?://\S+",
        "passport":    r"\b[A-Z]{1,2}\d{6,9}\b",
    }

    # name-based heuristics (column name level)
    name_keywords = {
        "full_name": ["name", "fullname", "full_name", "customer_name", "person"],
        "address":   ["address", "street", "city", "state", "postcode", "zipcode", "location"],
        "dob":       ["birth", "dob", "born", "birthday"],
        "national_id": ["ssn", "national_id", "passport", "license", "nid", "aadhaar", "pan"],
        "financial": ["account", "routing", "iban", "card_number", "cvv", "balance"],
    }

    for col in df.columns:
        col_findings = []
        clow = col.lower().replace(" ", "_")
        s = df[col].dropna().astype(str)

        # pattern matching
        for pii_type, pattern in pii_patterns.items():
            match_rate = s.str.contains(pattern, regex=True, na=False).mean()
            if match_rate > 0.3:
                # mask sample: show only first/last char with ***
                sample_raw = s.iloc[0] if len(s) else ""
                masked = sample_raw[:2] + "***" + sample_raw[-2:] if len(sample_raw) > 4 else "***"
                col_findings.append({
                    "pii_type": pii_type,
                    "match_rate_pct": round(match_rate * 100, 1),
                    "masked_sample": masked,
                })

        # column-name based detection
        for pii_type, keywords in name_keywords.items():
            if any(kw in clow for kw in keywords) and not col_findings:
                col_findings.append({
                    "pii_type": pii_type,
                    "detection_basis": "column_name_heuristic",
                    "note": "Verify manually — detected by column name, not value pattern",
                })

        if col_findings:
            findings[col] = col_findings

    anonymization_options = {
        "email": "hash (sha256) or replace with synthetic email",
        "phone": "mask last 7 digits: df[col].str.replace(r'\\d(?=\\d{7})', '*', regex=True)",
        "ssn_us": "drop column or replace entirely with pseudonymous ID",
        "full_name": "hash or replace with fake name using Faker library",
        "credit_card": "mask: keep first 4 and last 4 digits only",
        "ip_address": "generalize to /24 subnet or hash",
        "address": "drop street-level detail, keep only city/region",
        "dob": "replace with age bucket (0-18, 18-35, 35-50, 50+)",
    }

    return {
        "pii_findings": findings,
        "total_pii_columns": len(findings),
        "anonymization_options": anonymization_options,
        "warning": "GDPR/CCPA compliance may require removal or pseudonymization before sharing/training.",
    }


# ── 30. TEMPORAL DATA ANALYSIS  (Problem #15, #25, #26) ─────────────
def tool_analyze_temporal_issues(working_files: dict, filename: str, datetime_col: str) -> dict:
    """
    Deep temporal analysis for time-series or event-log data:
    - Irregular time intervals (gaps, bursts)
    - Timezone inconsistencies
    - Seasonality and trend signals
    - Temporal leakage risk (future data mixed into past periods)
    - Recommended chronological train/test split point
    - Data drift: distribution of numeric columns across time buckets
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    if datetime_col not in df.columns:
        return {"error": f"Column '{datetime_col}' not found"}

    ts = pd.to_datetime(df[datetime_col], errors="coerce")
    valid = ts.dropna().sort_values()
    if len(valid) < 10:
        return {"error": "Not enough parseable datetime values"}

    result: dict[str, Any] = {
        "datetime_col": datetime_col,
        "total_rows": len(df),
        "valid_timestamps": len(valid),
        "unparseable": int(ts.isnull().sum()),
        "min": str(valid.iloc[0]),
        "max": str(valid.iloc[-1]),
        "span_days": int((valid.iloc[-1] - valid.iloc[0]).days),
    }

    # timezone
    result["timezone_aware"] = bool(valid.dt.tz is not None)
    if not result["timezone_aware"]:
        result["timezone_warning"] = "No timezone info — if data comes from multiple regions, join errors possible"

    # gaps analysis
    diffs = valid.diff().dropna()
    median_gap = diffs.median()
    max_gap = diffs.max()
    gaps_5x = int((diffs > median_gap * 5).sum())
    result["median_interval"] = str(median_gap)
    result["max_gap"] = str(max_gap)
    result["large_gaps_5x_median"] = gaps_5x
    if gaps_5x > 0:
        result["gap_recommendation"] = "Irregular intervals detected — consider resampling to fixed frequency before time-series modeling"

    # seasonality signal (month distribution)
    month_counts = valid.dt.month.value_counts().sort_index().to_dict()
    month_cv = float(pd.Series(list(month_counts.values())).std() / max(pd.Series(list(month_counts.values())).mean(), 1))
    result["monthly_distribution"] = {str(k): v for k, v in month_counts.items()}
    if month_cv > 0.3:
        result["seasonality_signal"] = f"CV={round(month_cv,2)} — uneven monthly distribution, possible seasonality or data gaps"

    # chronological split recommendation
    split_80 = valid.quantile(0.8)
    result["recommended_train_test_split"] = {
        "train_end": str(split_80),
        "note": "Use chronological split — never random split for temporal data (temporal leakage)",
    }

    # data drift: compare first vs last quarter of numeric columns
    df_sorted = df.loc[ts.sort_values().index].reset_index(drop=True) if len(ts.dropna()) == len(df) else df
    n = len(df_sorted)
    q1_df = df_sorted.iloc[:n // 4]
    q4_df = df_sorted.iloc[3 * n // 4:]
    drift = {}
    for col in df_sorted.select_dtypes(include=np.number).columns:
        s1, s4 = q1_df[col].dropna(), q4_df[col].dropna()
        if len(s1) > 10 and len(s4) > 10:
            try:
                ks_stat, ks_p = scipy_stats.ks_2samp(s1.values, s4.values)
                if ks_p < 0.05:
                    drift[col] = {
                        "ks_statistic": round(float(ks_stat), 3),
                        "p_value": round(float(ks_p), 4),
                        "mean_q1": round(float(s1.mean()), 3),
                        "mean_q4": round(float(s4.mean()), 3),
                        "interpretation": "distribution_shifted — possible data drift",
                    }
            except Exception:
                pass

    if drift:
        result["data_drift_signals"] = drift

    return result


# ── 31. GRANULARITY MISMATCH  (Problem #32, #18) ────────────────────
def tool_check_granularity(working_files: dict) -> dict:
    """
    When multiple files are loaded, detect granularity mismatches:
    - Different row counts suggesting different aggregation levels
    - Duplicate key values in one file (summary) vs unique in another (detail)
    - Date column granularity (daily vs monthly vs yearly)
    Helps diagnose bad joins before they inflate or deflate row counts.
    """
    if len(working_files) < 2:
        return {"note": "Only one file loaded — granularity check requires multiple files"}

    summaries = {}
    for name, path in working_files.items():
        try:
            df = _load(path)
            dt_cols = list(df.select_dtypes(include="datetime64").columns)
            dt_granularity = {}
            for c in dt_cols:
                s = df[c].dropna()
                if len(s) == 0:
                    continue
                has_time   = (s.dt.hour != 0).any() or (s.dt.minute != 0).any()
                has_day    = (s.dt.day != 1).any()
                granularity = "timestamp" if has_time else ("daily" if has_day else "monthly_or_coarser")
                dt_granularity[c] = granularity

            # find likely join key candidates
            id_cols = [c for c in df.columns if any(k in c.lower() for k in ["id","key","code","uuid"])]
            key_info = {}
            for c in id_cols:
                key_info[c] = {
                    "unique_values": int(df[c].nunique()),
                    "total_rows": len(df),
                    "is_unique_key": bool(df[c].nunique() == len(df)),
                    "duplicate_count": int(df[c].duplicated().sum()),
                }

            summaries[name] = {
                "rows": len(df),
                "cols": len(df.columns),
                "datetime_granularity": dt_granularity,
                "key_column_info": key_info,
            }
        except Exception as e:
            summaries[name] = {"error": str(e)}

    # cross-file warnings
    warnings = []
    names = list(summaries.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = summaries[names[i]], summaries[names[j]]
            if "rows" in a and "rows" in b:
                ratio = max(a["rows"], b["rows"]) / max(min(a["rows"], b["rows"]), 1)
                if ratio > 10:
                    warnings.append(f"Row count ratio {names[i]}:{names[j]} = {round(ratio,1)}:1 — likely different granularity levels")
            # datetime granularity mismatch
            if "datetime_granularity" in a and "datetime_granularity" in b:
                grans_a = set(a["datetime_granularity"].values())
                grans_b = set(b["datetime_granularity"].values())
                if grans_a and grans_b and grans_a != grans_b:
                    warnings.append(f"Datetime granularity mismatch: {names[i]}={grans_a} vs {names[j]}={grans_b} — align before joining")

    return {"file_summaries": summaries, "cross_file_warnings": warnings}


# ── 32. LABEL QUALITY CHECK  (Problem #28) ──────────────────────────
def tool_check_label_quality(working_files: dict, filename: str, label_col: str) -> dict:
    """
    Assess quality of a target/label column for supervised learning:
    - Null labels (unusable rows)
    - Suspicious label patterns (all same, near-constant, impossible values)
    - For numeric targets: outlier labels, truncated distributions
    - For classification: class overlap heuristic using feature variance by class
    - Detects if labels look like they were auto-generated or bulk-assigned
    """
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    df = _load(path)
    if label_col not in df.columns:
        return {"error": f"Column '{label_col}' not found"}

    s = df[label_col]
    null_count = int(s.isnull().sum())
    nunique = int(s.nunique())
    total = len(s)
    issues = []

    if null_count > 0:
        issues.append(f"NULL_LABELS: {null_count} rows ({round(null_count/total*100,1)}%) have no label — cannot be used for training")

    if nunique == 1:
        issues.append("CONSTANT_LABEL: all rows have the same label — no signal to learn")
    elif nunique / total > 0.8 and not pd.api.types.is_numeric_dtype(s):
        issues.append("HIGH_CARDINALITY_LABEL: most labels are unique — may be an ID column, not a real target")

    # for numeric labels
    if pd.api.types.is_numeric_dtype(s):
        clean = s.dropna()
        # truncation detection: too many rows at exactly min or max
        for bound in [clean.min(), clean.max()]:
            pct_at_bound = (clean == bound).mean()
            if pct_at_bound > 0.05:
                issues.append(f"TRUNCATED_LABEL: {round(pct_at_bound*100,1)}% of labels == {bound} — may be a capped/clipped target")
        # outlier labels
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        outlier_labels = ((clean < q1 - 3*iqr) | (clean > q3 + 3*iqr)).sum()
        if outlier_labels > 0:
            issues.append(f"OUTLIER_LABELS: {outlier_labels} extreme label values — verify they are not data entry errors")

    # bulk-assignment detection: suspicious round numbers or patterns
    if not pd.api.types.is_numeric_dtype(s):
        vc = s.value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[0] > 0.95:
            issues.append(f"NEAR_CONSTANT_LABEL: {round(vc.iloc[0]*100,1)}% of rows have label='{vc.index[0]}' — extreme class imbalance")

    # class-feature separation (only for low-cardinality categorical)
    separation_info = {}
    if nunique <= 10 and nunique >= 2:
        num_cols = df.select_dtypes(include=np.number).columns.difference([label_col]).tolist()
        for col in num_cols[:5]:  # check first 5 numeric features
            try:
                by_class = df.groupby(label_col)[col].mean()
                spread = float(by_class.std())
                overall_std = float(df[col].std())
                separation = round(spread / overall_std, 3) if overall_std > 0 else 0
                separation_info[col] = separation
            except Exception:
                pass

    return {
        "label_col": label_col,
        "total_rows": total,
        "null_labels": null_count,
        "n_classes": nunique,
        "class_distribution": s.value_counts().head(10).to_dict(),
        "label_issues": issues,
        "feature_class_separation_scores": separation_info,
        "note": "Separation score > 0.5 means feature varies meaningfully across classes — good signal.",
    }


# ═══════════════════════════════════════════════════════════════════
#  TOOL REGISTRY
#  name → (callable, description, parameters_schema)
# ═══════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "profile_dataframe": (
        tool_profile_dataframe,
        "Overview of all files: dtypes, nulls, duplicates, flagged columns. Call FIRST.",
        {"type":"object","properties":{},"required":[]}
    ),
    "inspect_column": (
        tool_inspect_column,
        "Full forensic report on one column: stats, value counts, detected patterns.",
        {"type":"object","properties":{"filename":{"type":"string"},"col":{"type":"string"}},"required":["filename","col"]}
    ),
    "check_correlations": (
        tool_check_correlations,
        "Find highly correlated numeric pairs and near-duplicate text columns.",
        {"type":"object","properties":{"filename":{"type":"string"},"threshold":{"type":"number","default":0.95}},"required":["filename"]}
    ),
    "detect_schema_anomalies": (
        tool_detect_schema_anomalies,
        "Detect constant columns, mixed types, negative values in positive cols, PK violations.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_distributions": (
        tool_analyze_distributions,
        "Skewness, kurtosis, outlier counts, and transform recommendations for all numeric columns.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_missing": (
        tool_analyze_missing,
        "Null counts and imputation strategy recommendations per column.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_duplicates": (
        tool_analyze_duplicates,
        "Exact row duplicates, key-column duplicates, timestamp duplicates.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_categories": (
        tool_analyze_categories,
        "Value counts, case variants, rare categories, encoding recommendations for one categorical column.",
        {"type":"object","properties":{"filename":{"type":"string"},"col":{"type":"string"}},"required":["filename","col"]}
    ),
    "analyze_datetimes": (
        tool_analyze_datetimes,
        "Detect date columns, parse issues, future/ancient dates, feature engineering opportunities.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_text_quality": (
        tool_analyze_text_quality,
        "HTML tags, encoding issues, boilerplate, embedded JSON, length variance in text columns.",
        {"type":"object","properties":{"filename":{"type":"string"},"col":{"type":"string"}},"required":["filename","col"]}
    ),
    "check_cross_column_consistency": (
        tool_check_cross_column_consistency,
        "Logical inconsistencies: end<start dates, financial triangles, age/birthdate mismatches.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "detect_encoding_issues": (
        tool_detect_encoding_issues,
        "Mojibake, null bytes, RTL markers, zero-width chars in string columns.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "inspect_outliers": (
        tool_inspect_outliers,
        "IQR and z-score outlier bounds, extreme sample values, and fix recommendations for one column.",
        {"type":"object","properties":{"filename":{"type":"string"},"col":{"type":"string"}},"required":["filename","col"]}
    ),
    "suggest_feature_engineering": (
        tool_suggest_feature_engineering,
        "Date decomposition, ratio features, cyclical encoding, text length features suggestions.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "scan_data_leakage": (
        tool_scan_data_leakage,
        "Flag columns that may encode the target or cause leakage. Pass target_col if known.",
        {"type":"object","properties":{"filename":{"type":"string"},"target_col":{"type":"string"}},"required":["filename"]}
    ),
    "run_transformation": (
        tool_run_transformation,
        "Execute Python code to fix a data issue. Code must load via pd.read_pickle(path) and save with df.to_pickle(path).",
        {"type":"object","properties":{"code":{"type":"string"},"description":{"type":"string"}},"required":["code","description"]}
    ),
    "verify_result": (
        tool_verify_result,
        "Audit the dataframe after a transformation: row/col counts, null diffs, dtype changes.",
        {"type":"object","properties":{"filename":{"type":"string"},"checks":{"type":"array","items":{"type":"string"}}},"required":["filename"]}
    ),
    "get_sample_rows": (
        tool_get_sample_rows,
        "Return n sample rows, optionally filtered by a pandas query string.",
        {"type":"object","properties":{"filename":{"type":"string"},"n":{"type":"integer","default":10},"condition":{"type":"string"}},"required":["filename"]}
    ),
    "compare_before_after": (
        tool_compare_before_after,
        "Compare a column's stats before and after the last transformation.",
        {"type":"object","properties":{"filename":{"type":"string"},"col":{"type":"string"}},"required":["filename","col"]}
    ),
    "request_human_input": (
        tool_request_human_input,
        "Ask the human a domain question that cannot be answered from data alone.",
        {"type":"object","properties":{"question":{"type":"string"}},"required":["question"]}
    ),
    "detect_fuzzy_duplicates": (
        tool_detect_fuzzy_duplicates,
        "Find near-duplicate string values in a column using edit-distance (e.g. 'USA' vs 'US', name typos). Use on name/address/category columns after exact dedup.",
        {"type":"object","properties":{"filename":{"type":"string"},"col":{"type":"string"},"threshold":{"type":"integer","default":90}},"required":["filename","col"]}
    ),
    "validate_ranges": (
        tool_validate_ranges,
        "Detect domain-rule violations: negative ages, prices, counts; out-of-range percentages; suspiciously frequent sentinel values (-1, 9999, 0).",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "check_multicollinearity": (
        tool_check_multicollinearity,
        "Compute VIF scores for all numeric columns. VIF>5=moderate, VIF>10=severe multicollinearity. Also returns high-correlation pairs (>0.7).",
        {"type":"object","properties":{"filename":{"type":"string"},"vif_threshold":{"type":"number","default":5.0}},"required":["filename"]}
    ),
    "analyze_scaling": (
        tool_analyze_scaling,
        "Check if numeric columns need scaling. Reports range, magnitude ratio across columns, detects already-scaled columns, recommends StandardScaler vs RobustScaler vs MinMax.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "detect_structural_issues": (
        tool_detect_structural_issues,
        "Find structural data problems: bad column names (spaces, special chars), multi-value packed columns, wide-format data that should be melted.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "detect_unit_inconsistencies": (
        tool_detect_unit_inconsistencies,
        "Find columns with mixed measurement units: kg/lbs, Celsius/Fahrenheit, km/miles, mixed currencies. Returns bimodal distribution evidence.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_sparsity": (
        tool_analyze_sparsity,
        "Measure zero-fill and null-fill rates. Flags sparse columns, binary one-hot columns, and recommends sparse storage or dimensionality reduction.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_class_imbalance": (
        tool_analyze_class_imbalance,
        "Analyse class distribution for a target column. Returns imbalance ratio and recommends SMOTE, class_weight, or threshold tuning.",
        {"type":"object","properties":{"filename":{"type":"string"},"target_col":{"type":"string"}},"required":["filename","target_col"]}
    ),
    "scan_pii": (
        tool_scan_pii,
        "Detect PII columns: emails, phones, SSNs, credit cards, names, addresses, DOB. Returns masked samples and anonymization options. NEVER returns raw PII.",
        {"type":"object","properties":{"filename":{"type":"string"}},"required":["filename"]}
    ),
    "analyze_temporal_issues": (
        tool_analyze_temporal_issues,
        "Deep time-series analysis: irregular intervals, timezone issues, seasonality, data drift (KS test first vs last quarter), chronological split recommendation.",
        {"type":"object","properties":{"filename":{"type":"string"},"datetime_col":{"type":"string"}},"required":["filename","datetime_col"]}
    ),
    "check_granularity": (
        tool_check_granularity,
        "When multiple files are loaded: detect row-count ratio mismatches and datetime granularity differences (daily vs monthly) that cause bad joins.",
        {"type":"object","properties":{},"required":[]}
    ),
    "check_label_quality": (
        tool_check_label_quality,
        "Assess label/target column quality: null labels, constant labels, truncated distributions, outlier labels, class-feature separation scores.",
        {"type":"object","properties":{"filename":{"type":"string"},"label_col":{"type":"string"}},"required":["filename","label_col"]}
    ),
}


def _build_tool_schemas() -> list[dict]:
    schemas = []
    for name, (_, description, params) in TOOL_REGISTRY.items():
        schemas.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": params}
        })
    return schemas


TOOL_SCHEMAS = _build_tool_schemas()


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — small and stable
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior data analyst and data engineer. You have been given a set of dataframes (as pickle files) to clean and prepare for analysis or machine learning.

Your job is to find and fix ALL data quality issues using the available tools. Think like a thorough data analyst.

WORKFLOW:
1. profile_dataframe() — overview of all files (always start here)
2. If multiple files: check_granularity() — detect join/merge level mismatches
3. Use targeted inspection tools for each flagged column or issue
4. run_transformation() to fix — write real Python, not pseudo-code
5. verify_result() + compare_before_after() after every fix
6. Repeat until profile_dataframe() shows no remaining issues

PROBLEM CHECKLIST — inspect each category before declaring done:
• Missing: analyze_missing() → impute (KNN, median/groupby, mode, 'Unknown') — NEVER dropna()
• Duplicates: analyze_duplicates() → exact rows; detect_fuzzy_duplicates() for near-matches in key text cols
• Types: inspect_column() flags numeric-as-string, date-as-object, bool-as-int → cast appropriately
• Outliers: analyze_distributions() + inspect_outliers() → clip / log1p / yeo-johnson / flag
• Range violations: validate_ranges() → negative ages, impossible percentages, sentinel values
• Categories: analyze_categories() → case variants, whitespace, rare → standardize + group_rare
• Dates: analyze_datetimes() → parse strings, fix timezone, engineer features; analyze_temporal_issues() for time-series
• Text: analyze_text_quality() → HTML, encoding, boilerplate; detect_encoding_issues() for mojibake
• Structure: detect_structural_issues() → bad column names, packed multi-values, wide→long reshape
• Units: detect_unit_inconsistencies() → mixed kg/lbs, °C/°F, currencies
• Multicollinearity: check_multicollinearity() → VIF scores; drop or combine redundant features
• Scaling: analyze_scaling() → flag when magnitude ratio >100×; recommend scaler type
• Sparsity: analyze_sparsity() → sparse features, high-zero columns
• Class imbalance: analyze_class_imbalance() → imbalance ratio; recommend SMOTE/class_weight
• PII: scan_pii() → mask or anonymize sensitive columns before use
• Leakage: scan_data_leakage() + analyze_temporal_issues() → temporal leakage, target-correlated cols
• Label quality: check_label_quality() → null labels, truncated targets, class separation
• Cross-column logic: check_cross_column_consistency() → end<start, financial triangles
• Correlations: check_correlations() + check_multicollinearity() → redundant columns

TRANSFORMATION RULES:
- NEVER drop rows with dropna() — always impute, flag, or keep
- Deduplication (drop_duplicates) is the only allowed row removal
- SMOTE must only be applied after train/test split — flag it as a recommendation, do not apply to full dataset
- Scaling parameters must be fit on training data only — flag if split not yet done
- Always save with df.to_pickle(path) to the SAME path
- After each fix: verify_result() to confirm, then continue

When done: say "CLEANING COMPLETE" and summarise every change made."""


# ═══════════════════════════════════════════════════════════════════
#  TOOL DISPATCHER
# ═══════════════════════════════════════════════════════════════════

def dispatch_tool(tool_name: str, tool_args: dict, working_files: dict) -> str:
    """Route a tool call to its implementation and return JSON result."""
    if tool_name not in TOOL_REGISTRY:
        return _j({"error": f"Unknown tool: {tool_name}. Available: {list(TOOL_REGISTRY.keys())}"})

    fn, _, _ = TOOL_REGISTRY[tool_name]
    import inspect
    sig = inspect.signature(fn)
    kwargs = dict(tool_args)

    # inject working_files for tools that need it
    if "working_files" in sig.parameters:
        kwargs["working_files"] = working_files

    try:
        result = fn(**kwargs)
        return _trunc(_j(result))
    except TypeError as e:
        return _j({"error": f"Tool call error: {e}"})
    except Exception as e:
        return _j({"error": f"Tool execution failed: {traceback.format_exc()}"})


# ═══════════════════════════════════════════════════════════════════
#  LANGGRAPH NODES
# ═══════════════════════════════════════════════════════════════════

def analyst_agent_node(state: MasterState) -> dict:
    """
    Core ReAct agent node. Sends conversation history to the LLM
    with tools. Returns the AI message (may contain tool calls).
    """
    messages = state.get("messages", [])
    log_entries = list(state.get("agent_log", []))

    # First call: inject system prompt + user instruction
    if not messages:
        user_instruction = state.get("user_input", "Please clean and prepare this dataset.")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User instruction: {user_instruction}\n\nworking_files: {_j(state.get('working_files', {}))}"),
        ]

    llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    log_entries.append(make_log_entry(
        "Cleaning", "Reviewing dataset state",
        state.get("user_input", "Cleaning pass in progress.")[:120], "running",
    ))

    return {
        "messages": messages,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "error": None,
        "agent_log": log_entries,
        "active_agent": "cleaning",
        "pipeline_status": "running",
    }


def tool_executor_node(state: MasterState) -> dict:
    """
    Execute all tool calls from the last AI message.
    Appends ToolMessage results back to the conversation.
    Handles the special __human_question__ signal.
    """
    messages  = state.get("messages", [])
    last_ai   = messages[-1]
    working_files = state.get("working_files", {})
    new_messages  = list(messages)
    pending_question = None

    for tc in last_ai.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id   = tc["id"]

        raw_result = dispatch_tool(tool_name, tool_args, working_files)
        result_obj = json.loads(raw_result) if raw_result.startswith("{") or raw_result.startswith("[") else {"output": raw_result}

        # check for human escalation
        if isinstance(result_obj, dict) and "__human_question__" in result_obj:
            pending_question = result_obj["__human_question__"]
            new_messages.append(ToolMessage(content=_j({"status": "awaiting_human_input"}), tool_call_id=tool_id))
            break

        new_messages.append(ToolMessage(content=raw_result, tool_call_id=tool_id))

    return {
        "messages": new_messages,
        "pending_human_question": pending_question,
        "active_agent": "cleaning",
        "pipeline_status": "running",
    }


async def human_review_node(state: MasterState, config: RunnableConfig) -> dict:
    """Interrupt for human input — either domain question or final approval."""
    question = state.get("pending_human_question") or \
        "Agent has completed a pass. Type 'approve' to finalize or provide additional instructions."

    # Auto-approve: skip interrupt and auto-approve
    if AUTO_APPROVE:
        import logging
        logging.getLogger(__name__).info("AUTO_APPROVE: auto-approving clean agent review")
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content="Human response: approve"))
        return {
            "messages": messages,
            "user_feedback": "approve",
            "pending_human_question": None,
            "iteration_count": 0,
            "active_agent": "cleaning",
            "pipeline_status": "running",
        }

    feedback = interrupt(question)
    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=f"Human response: {feedback}"))
    return {
        "messages": messages,
        "user_feedback": feedback,
        "pending_human_question": None,
        "iteration_count": 0,
        "active_agent": "cleaning",
        "pipeline_status": "waiting",
    }


# ═══════════════════════════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════════════════════════

def route_agent(state: MasterState) -> str:
    """After agent responds: go to tools, human, or end."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last = messages[-1]
    # hard stop
    if state.get("iteration_count", 0) >= MAX_TOOL_TURNS:
        return "human_review"

    # if AI called tools → execute them
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_executor"

    # if AI said cleaning is complete → human final approval
    content = getattr(last, "content", "") or ""
    if "CLEANING COMPLETE" in content.upper():
        return "human_review"

    # AI responded with plain text but no tool call → nudge it back (loop continues)
    # human_review only when CLEANING COMPLETE or hard stop above
    return "analyst_agent"


def route_after_tools(state: MasterState) -> str:
    """After tools execute: back to agent, or pause for human if question pending."""
    if state.get("pending_human_question"):
        return "human_review"
    return "analyst_agent"


def route_after_human(state: MasterState) -> str:
    """After human responds: end if approved, else continue agent."""
    feedback = str(state.get("user_feedback") or "").strip().lower()
    if feedback in ["approve", "yes", "ok", "done", "finish", "complete", ""]:
        return END
    return "analyst_agent"


# ═══════════════════════════════════════════════════════════════════
#  GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════

def build_cleaning_graph():
    workflow = StateGraph(MasterState)

    workflow.add_node("analyst_agent",  analyst_agent_node)
    workflow.add_node("tool_executor",  tool_executor_node)
    workflow.add_node("human_review",   human_review_node)

    workflow.set_entry_point("analyst_agent")

    workflow.add_conditional_edges("analyst_agent", route_agent, {
        "analyst_agent": "analyst_agent",
        "tool_executor": "tool_executor",
        "human_review":  "human_review",
        END:              END,
    })

    workflow.add_conditional_edges("tool_executor", route_after_tools, {
        "analyst_agent": "analyst_agent",
        "human_review":  "human_review",
    })

    workflow.add_conditional_edges("human_review", route_after_human, {
        "analyst_agent": "analyst_agent",
        END:              END,
    })

    return workflow
