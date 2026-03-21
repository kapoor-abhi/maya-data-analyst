"""
ml/ml_agent.py — Intelligent ML Training & Prediction Agent

ReAct loop with tools for:
- Inspecting data for ML readiness
- Auto-training with model selection (XGBoost, LightGBM, sklearn)
- Running CUSTOM ML code for any model type (Prophet, LSTM, SVM, etc.)
- Predictions on new data

BUG FIX: route_ml_agent now correctly continues the ReAct loop instead of
always jumping to review after the first message.
"""
import os
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.runnables.config import RunnableConfig

from core.state import MasterState
from core.llm import get_llm
from core.sandbox import DockerREPL, _strip_code
from core.activity_log import make_log_entry

load_dotenv()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

AUTO_APPROVE = os.getenv("AUTO_APPROVE", "").lower() in ("1", "true", "yes")

coder_llm    = get_llm("coder", temperature=0.0)
MAX_ML_TURNS = 20


def _load(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def _trunc(s: str, n: int = 4000) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


# ── ML Tools ──────────────────────────────────────────────────────────

def tool_inspect_for_ml(working_files: dict) -> dict:
    """Inspect dataset for ML readiness: dtypes, null counts, class distribution."""
    result = {}
    for name, path in working_files.items():
        df = _load(path)
        # Class distribution for potential target columns
        target_hints = {}
        for col in df.select_dtypes(include=["object", "category", "int64", "float64"]).columns[:5]:
            vc = df[col].value_counts()
            if len(vc) <= 30:
                target_hints[col] = {"value_counts": vc.head(10).to_dict(),
                                     "nunique": int(df[col].nunique())}

        result[name] = {
            "shape":           list(df.shape),
            "columns":         {c: str(t) for c, t in df.dtypes.items()},
            "null_counts":     {c: int(df[c].isnull().sum())
                                for c in df.columns if df[c].isnull().any()},
            "numeric_cols":    df.select_dtypes(include=np.number).columns.tolist(),
            "datetime_cols":   df.select_dtypes(include="datetime64").columns.tolist(),
            "categorical_cols":df.select_dtypes(include="object").columns.tolist(),
            "potential_targets":target_hints,
            "sample_rows":     df.head(3).to_dict(orient="records"),
        }
    return result


def tool_auto_train(
    working_files: dict,
    filename: str,
    target_col: str,
    task_type: str,          # "regression" | "classification" | "forecast"
    model_save_path: str,
    feature_cols: Optional[list] = None,
    datetime_col: Optional[str]  = None,
    forecast_periods: int = 30,
    model_preference: Optional[str] = None,  # "xgboost"|"lightgbm"|"random_forest"|"linear"|"auto"
) -> dict:
    """Auto-select model, train with CV + hold-out split, save. Returns metrics + feature importance."""
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}

    if not model_save_path.endswith(".pkl"):
        model_save_path = os.path.join(model_save_path, "trained_model.pkl")

    df = _load(path)
    if target_col not in df.columns:
        return {"error": f"Target '{target_col}' not found. Available: {list(df.columns)}"}

    # ── Feature selection
    if feature_cols:
        available = [c for c in feature_cols if c in df.columns]
    else:
        exclude = {target_col}
        if datetime_col:
            exclude.add(datetime_col)
        available = [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in exclude
            and not any(kw in c.lower() for kw in ["id", "uuid", "key", "index"])
        ]

    if not available:
        return {"error": "No usable numeric feature columns. Run feature_engineering first."}

    work = df[available + [target_col]].copy()
    if datetime_col and datetime_col in df.columns:
        work[datetime_col] = df[datetime_col]

    work = work.dropna(subset=[target_col])
    for col in available:
        if work[col].isnull().any():
            work[col] = work[col].fillna(work[col].median())
    for col in work[available].select_dtypes(include="object").columns:
        work[col] = pd.Categorical(work[col]).codes

    X = work[available].values.astype(float)
    y = work[target_col].values

    # Robust NaN/inf handling
    X = np.where(np.isinf(X), np.nan, X)
    col_med = np.where(np.isnan(np.nanmedian(X, axis=0)), 0, np.nanmedian(X, axis=0))
    for ci in range(X.shape[1]):
        X[np.isnan(X[:, ci]), ci] = col_med[ci]
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y.astype(float))
    X, y = X[valid], y[valid]

    if len(X) < 10:
        return {"error": f"Only {len(X)} valid rows — need at least 10"}

    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

    le = None
    if task_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ── Train/test split for honest hold-out evaluation
    is_ts = task_type == "forecast" or (datetime_col and task_type == "regression")
    if is_ts:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        cv = TimeSeriesSplit(n_splits=min(5, len(X_train) // 10))
        scoring = "neg_mean_absolute_error"
    elif task_type == "regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "neg_mean_absolute_error"
    else:
        n_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if n_classes > 1 else None)
        cv = StratifiedKFold(n_splits=min(5, len(X_train) // max(n_classes, 1)),
                             shuffle=True, random_state=42)
        scoring = "f1_weighted"

    # ── Model candidates
    def _get_models(task: str, pref: Optional[str]):
        models = {}
        # Filter by preference if given
        want_all = (not pref) or pref == "auto"

        if want_all or pref == "xgboost":
            try:
                import xgboost as xgb
                if task == "regression":
                    models["XGBoost"] = xgb.XGBRegressor(
                        n_estimators=200, learning_rate=0.1, max_depth=6,
                        random_state=42, verbosity=0)
                else:
                    models["XGBoost"] = xgb.XGBClassifier(
                        n_estimators=200, learning_rate=0.1, max_depth=6,
                        random_state=42, verbosity=0, eval_metric="logloss")
            except ImportError:
                pass

        if want_all or pref == "lightgbm":
            try:
                import lightgbm as lgb
                if task == "regression":
                    models["LightGBM"] = lgb.LGBMRegressor(
                        n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1)
                else:
                    models["LightGBM"] = lgb.LGBMClassifier(
                        n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1)
            except ImportError:
                pass

        if want_all or pref == "random_forest":
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
            if task == "regression":
                models["RandomForest"] = RandomForestRegressor(
                    n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
                if want_all:
                    models["ExtraTrees"] = ExtraTreesRegressor(
                        n_estimators=150, random_state=42, n_jobs=-1)
            else:
                models["RandomForest"] = RandomForestClassifier(
                    n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
                if want_all:
                    models["ExtraTrees"] = ExtraTreesClassifier(
                        n_estimators=150, random_state=42, n_jobs=-1)

        if want_all or pref == "linear":
            from sklearn.linear_model import Ridge, LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            if task == "regression":
                models["Ridge"] = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0))
                ])
            else:
                models["LogisticRegression"] = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, random_state=42))
                ])

        if not models:  # fallback
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            if task == "regression":
                models["GradientBoosting"] = GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
            else:
                models["GradientBoosting"] = GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)

        return models

    ml_task = "regression" if task_type in ["regression", "forecast"] else "classification"
    models  = _get_models(ml_task, model_preference)

    best_name, best_model, best_score = None, None, -np.inf
    cv_results = {}
    for name, model in models.items():
        try:
            scores     = cross_val_score(model, X_train, y_train, cv=cv,
                                          scoring=scoring, n_jobs=-1)
            mean_score = float(np.mean(scores))
            cv_results[name] = {"cv_mean": round(mean_score, 4),
                                 "cv_std":  round(float(np.std(scores)), 4)}
            if mean_score > best_score:
                best_score = mean_score
                best_name  = name
                best_model = model
        except Exception as e:
            cv_results[name] = {"error": str(e)}

    if best_model is None:
        return {"error": "All models failed cross-validation"}

    # Final fit on full training set
    best_model.fit(X_train, y_train)

    # Hold-out evaluation
    y_pred_test = best_model.predict(X_test)
    if ml_task == "regression":
        mae = round(float(mean_absolute_error(y_test, y_pred_test)), 4)
        r2 = round(float(r2_score(y_test, y_pred_test)), 4)
        metrics = {
            "MAE": mae,
            "MAE (test)": mae,
            "R²": r2,
            "R² (test)": r2,
        }
        # MAPE if positive
        if np.all(np.array(y_test, dtype=float) > 0):
            mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test))) * 100
            metrics["MAPE"] = round(mape, 2)
            metrics["MAPE (test)"] = round(mape, 2)
    else:
        accuracy = round(float(accuracy_score(y_test, y_pred_test)), 4)
        f1_weighted = round(float(f1_score(y_test, y_pred_test,
                                           average="weighted",
                                           zero_division=0)), 4)
        metrics = {
            "accuracy": accuracy,
            "accuracy (test)": accuracy,
            "f1_weighted": f1_weighted,
            "f1_weighted (test)": f1_weighted,
        }
    metrics["best_model"] = best_name

    # Re-fit on ALL data for final model
    best_model.fit(X, y)

    # Feature importance
    importance = {}
    last_step = best_model
    if hasattr(best_model, "named_steps"):
        last_step = list(best_model.named_steps.values())[-1]
    if hasattr(last_step, "feature_importances_"):
        importance = dict(sorted(
            zip(available, last_step.feature_importances_.tolist()),
            key=lambda x: -x[1],
        ))
        importance = {k: round(float(v), 4) for k, v in list(importance.items())[:20]}
    elif hasattr(last_step, "coef_"):
        coefs = last_step.coef_.flatten() if last_step.coef_.ndim > 1 else last_step.coef_
        importance = dict(sorted(
            zip(available, np.abs(coefs).tolist()),
            key=lambda x: -x[1],
        ))
        importance = {k: round(float(v), 4) for k, v in list(importance.items())[:20]}

    # Forecast future periods
    predictions_preview = None
    forecast_csv_path   = None
    if task_type == "forecast" and datetime_col and datetime_col in df.columns:
        last_date    = pd.to_datetime(df[datetime_col]).max()
        future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq="D")[1:]
        last_features = X[-1:].copy()
        future_preds  = []
        for i in range(forecast_periods):
            pred = float(best_model.predict(last_features)[0])
            future_preds.append({"date": str(future_dates[i].date()), "predicted": round(pred, 2)})
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        forecast_df       = pd.DataFrame(future_preds)
        predictions_preview = future_preds[:10]
        forecast_csv_path   = model_save_path.replace(".pkl", "_forecast.csv")
        forecast_df.to_csv(forecast_csv_path, index=False)
    else:
        y_pred_sample = best_model.predict(X[:5])
        predictions_preview = [
            {"actual": round(float(a), 3), "predicted": round(float(p), 3)}
            for a, p in zip(y[:5], y_pred_sample)
        ]

    # Save model artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    artifact = {
        "model":              best_model,
        "feature_cols":       available,
        "target_col":         target_col,
        "task_type":          task_type,
        "label_encoder":      le,
        "feature_importance": importance,
        "metrics":            metrics,
    }
    with open(model_save_path, "wb") as f:
        pickle.dump(artifact, f)

    return {
        "metrics":              metrics,
        "cv_results":           cv_results,
        "best_model":           best_name,
        "feature_importance":   importance,
        "predictions_preview":  predictions_preview,
        "model_saved_to":       model_save_path,
        "forecast_csv":         forecast_csv_path,
        "n_features":           len(available),
        "n_samples":            len(X),
    }


def tool_predict(working_files: dict, filename: str,
                 model_path: str, output_col: str = "prediction") -> dict:
    """Run saved model on new data and add prediction column."""
    path = working_files.get(filename)
    if not path:
        return {"error": f"File '{filename}' not found"}
    if not os.path.exists(model_path):
        return {"error": f"Model not found at {model_path}"}

    df = _load(path)
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    feature_cols = artifact["feature_cols"]
    model        = artifact["model"]
    le           = artifact.get("label_encoder")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Missing feature columns: {missing}"}

    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(X.median())

    preds = model.predict(X.values)
    if le is not None:
        preds = le.inverse_transform(preds.astype(int))

    df[output_col] = preds
    df.to_pickle(path)

    return {
        "status":              "success",
        "predictions_added_as": output_col,
        "sample":              df[[output_col]].head(10).to_dict(orient="records"),
    }


def tool_run_custom_ml_code(working_files: dict, user_id: str, code: str) -> dict:
    """
    Run completely custom ML code (Prophet, ARIMA, LSTM, SVM, stacking, etc.)
    in the sandbox. The code can use any library and must print its results.
    Results and any saved files are captured and returned.
    """
    sandbox = DockerREPL()
    code = _strip_code(code)

    model_dir = os.path.join("storage", user_id, "models")
    preamble = (
        f"import pandas as pd\nimport numpy as np\nimport os\n"
        f"working_files = {json.dumps(working_files)}\n"
        f"model_dir = '{model_dir}'\n"
        f"os.makedirs(model_dir, exist_ok=True)\n"
    )

    result = sandbox.run(preamble + code)
    if result.get("error"):
        return {"status": "error", "error": result["error"]}

    output = result.get("output", "")
    return {
        "status": "success",
        "output": output[:3000],
        "model_dir": model_dir,
    }


# ── Tool schemas ──────────────────────────────────────────────────────

ML_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_for_ml",
            "description": "Inspect dataset for ML readiness. Always call first.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_train",
            "description": (
                "Auto-select best model, train with CV + hold-out split, save model. "
                "Use for: XGBoost, LightGBM, RandomForest, linear models. "
                "For Prophet/ARIMA/LSTM/neural networks, use run_custom_ml_code instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename":         {"type": "string"},
                    "target_col":       {"type": "string"},
                    "task_type":        {"type": "string", "enum": ["regression", "classification", "forecast"]},
                    "model_save_path":  {"type": "string"},
                    "feature_cols":     {"type": "array", "items": {"type": "string"}},
                    "datetime_col":     {"type": "string"},
                    "forecast_periods": {"type": "integer", "default": 30},
                    "model_preference": {
                        "type": "string",
                        "enum": ["auto", "xgboost", "lightgbm", "random_forest", "linear"],
                        "description": "Force a specific model family (optional)",
                    },
                },
                "required": ["filename", "target_col", "task_type", "model_save_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_custom_ml_code",
            "description": (
                "Execute CUSTOM ML code in the sandbox. Use for: Prophet (time series), "
                "ARIMA, LSTM, neural networks, SVM, stacking ensembles, or any model "
                "that auto_train doesn't support. "
                "Code can use any library. Must print results clearly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python code. working_files and model_dir are pre-defined. "
                            "Load data with pd.read_pickle(working_files[name]). "
                            "Print all metrics and key results."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict",
            "description": "Apply a saved model to data and add prediction column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename":   {"type": "string"},
                    "model_path": {"type": "string"},
                    "output_col": {"type": "string", "default": "prediction"},
                },
                "required": ["filename", "model_path"],
            },
        },
    },
]

ML_SYSTEM = """You are a senior machine learning engineer.

Given a dataset and a business goal, you will:
1. Call inspect_for_ml() to understand the data structure
2. Identify the task type:
   - "regression"     → predict a number (price, sales, revenue)
   - "classification" → predict a category (churn yes/no, fraud, approval)
   - "forecast"       → predict future time periods (next month's sales)
3. Choose the right approach:
   - Standard models (XGBoost, LightGBM, RandomForest, linear): use auto_train()
   - Prophet for time-series with seasonality: use run_custom_ml_code()
   - ARIMA for stationary time-series: use run_custom_ml_code()
   - Neural networks / deep learning: use run_custom_ml_code()
   - Any exotic model the user requests: use run_custom_ml_code()
4. Report results in plain business language with feature importance interpretation

IMPORTANT RULES:
- "predict next month's sales" → task_type="forecast", include datetime_col
- "predict customer churn" → task_type="classification"
- "predict house price" → task_type="regression"
- "forecast with seasonality" → use run_custom_ml_code() with Prophet
- model_save_path is provided in the initial message
- Explain feature importance in plain English (not technical terms)
- Say "ML TRAINING COMPLETE" when done — include a concise results summary
"""


def dispatch_ml_tool(name: str, args: dict, working_files: dict, user_id: str) -> str:
    if name == "inspect_for_ml":
        result = tool_inspect_for_ml(working_files)
    elif name == "auto_train":
        result = tool_auto_train(
            working_files,
            args.get("filename", next(iter(working_files), "")),
            args.get("target_col", ""),
            args.get("task_type", "regression"),
            args.get("model_save_path", "model.pkl"),
            args.get("feature_cols"),
            args.get("datetime_col"),
            args.get("forecast_periods", 30),
            args.get("model_preference"),
        )
    elif name == "run_custom_ml_code":
        result = tool_run_custom_ml_code(working_files, user_id, args.get("code", ""))
    elif name == "predict":
        result = tool_predict(
            working_files,
            args.get("filename", ""),
            args.get("model_path", ""),
            args.get("output_col", "prediction"),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    try:
        s = json.dumps(result, default=str, indent=2)
        return _trunc(s)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Graph nodes ───────────────────────────────────────────────────────

def ml_agent_node(state: MasterState) -> dict:
    working_files    = state.get("working_files", {})
    user_instruction = state.get("user_input", "Train a predictive model")
    messages         = list(state.get("messages", []))
    log_entries      = list(state.get("agent_log", []))

    user_id    = state.get("user_id", "default")
    model_dir  = os.path.join("storage", user_id, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "trained_model.pkl")

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [
            SystemMessage(content=ML_SYSTEM),
            HumanMessage(content=(
                f"Goal: {user_instruction}\n"
                f"Files: {list(working_files.keys())}\n"
                f"Save model to: {model_path}\n"
                f"Start with inspect_for_ml() to understand the data."
            )),
        ]

    llm_with_tools = coder_llm.bind_tools(ML_TOOL_SCHEMAS)
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            log_entries.append(make_log_entry(
                "ML", f"Calling {tc['name']}",
                str(tc.get("args", {}))[:200], "running",
            ))

    return {
        "messages":        messages,
        "ml_model_path":   model_path,
        "ml_task_type":    state.get("ml_task_type"),
        "iteration_count": state.get("iteration_count", 0) + 1,
        "error":           None,
        "agent_log":       log_entries,
        "active_agent":    "ml",
    }


def ml_tool_executor_node(state: MasterState) -> dict:
    messages      = list(state.get("messages", []))
    working_files = state.get("working_files", {})
    user_id       = state.get("user_id", "default")
    log_entries   = list(state.get("agent_log", []))
    last          = messages[-1]
    ml_report     = state.get("ml_report")

    for tc in last.tool_calls:
        result_str = dispatch_ml_tool(tc["name"], tc["args"], working_files, user_id)
        messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

        # Capture training report
        if tc["name"] == "auto_train":
            try:
                report_obj = json.loads(result_str)
                if "metrics" in report_obj:
                    ml_report = result_str
                    log_entries.append(make_log_entry(
                        "ML", "Training complete",
                        str(report_obj.get("metrics", {})), "success",
                        {"metrics": report_obj.get("metrics", {}),
                         "model": report_obj.get("best_model")},
                    ))
            except Exception:
                pass
        elif tc["name"] == "run_custom_ml_code":
            try:
                r = json.loads(result_str)
                if r.get("status") == "success":
                    log_entries.append(make_log_entry(
                        "ML", "Custom code executed",
                        r.get("output", "")[:200], "success",
                    ))
            except Exception:
                pass

    return {"messages": messages, "ml_report": ml_report, "agent_log": log_entries}


async def ml_review_node(state: MasterState, config: RunnableConfig) -> dict:
    """Human review checkpoint after ML training."""
    messages     = state.get("messages", [])
    last_content = messages[-1].content if messages else ""
    ml_report_str = state.get("ml_report", "")

    summary = last_content
    if ml_report_str:
        try:
            r = json.loads(ml_report_str)
            summary += (
                f"\n\n**Model:** {r.get('best_model')}\n"
                f"**Metrics:** {r.get('metrics')}\n"
                f"**Top Features:** {list(r.get('feature_importance', {}).keys())[:5]}"
            )
            if r.get("forecast_csv"):
                summary += f"\n**Forecast saved to:** {r['forecast_csv']}"
        except Exception:
            pass

    if AUTO_APPROVE:
        logger.info("AUTO_APPROVE: auto-approving ML results")
        return {"user_feedback": "approve", "error": None, "iteration_count": 0}

    feedback = interrupt(
        f"🤖 **ML Training Complete!**\n\n{summary}\n\n"
        f"Type **'approve'** to continue, or request changes "
        f"(e.g. 'try a neural network' or 'add more features')."
    )
    return {"user_feedback": feedback, "error": None, "iteration_count": 0}


# ── Routing ───────────────────────────────────────────────────────────

def route_ml_agent(state: MasterState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return END
    last    = messages[-1]
    content = getattr(last, "content", "") or ""

    if state.get("iteration_count", 0) >= MAX_ML_TURNS:
        return "ml_review"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "ml_tools"
    if "ML TRAINING COMPLETE" in content.upper():
        return "ml_review"

    # ★ BUG FIX: was always returning "ml_review" here — broke the ReAct loop entirely.
    # Continue the loop so the agent can call more tools.
    return "ml_agent"


def route_after_ml_review(state: MasterState) -> str:
    feedback = str(state.get("user_feedback", "")).strip().lower()
    if feedback in ["approve", "yes", "ok", "done", ""]:
        return END
    return "ml_agent"


def build_ml_graph():
    workflow = StateGraph(MasterState)
    workflow.add_node("ml_agent",  ml_agent_node)
    workflow.add_node("ml_tools",  ml_tool_executor_node)
    workflow.add_node("ml_review", ml_review_node)

    workflow.set_entry_point("ml_agent")
    workflow.add_conditional_edges("ml_agent", route_ml_agent,
                                   {"ml_tools":  "ml_tools",
                                    "ml_review": "ml_review",
                                    "ml_agent":  "ml_agent",
                                    END: END})
    workflow.add_edge("ml_tools", "ml_agent")
    workflow.add_conditional_edges("ml_review", route_after_ml_review,
                                   {"ml_agent": "ml_agent", END: END})
    return workflow
