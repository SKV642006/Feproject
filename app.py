"""
Credit Risk Platform - Flask Backend
=====================================
Full ML pipeline: Upload → Cleaning → WoE/IV → Modeling → Scorecard → Monitoring
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_file
)
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from typing import Tuple

warnings.filterwarnings("ignore")

# ============================================================
# APP CONFIGURATION
# ============================================================
app = Flask(__name__)
app.secret_key = "credit_risk_secret_key_2024"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# ============================================================
# GLOBAL STATE
# ============================================================
state = {
    "train_df": None,
    "cleaned_df": None,
    "target_col": None,
    "iv_summary": None,
    "binning_result": None,
    "feature_bins": None,
    "woe_mappings": None,
    "selected_features": None,
    "model": None,
    "scaler": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "train_scores": None,
    "test_scores": None,
    "scorecard_params": None,
    "monitoring_df": None,
    "model_metrics": None,
    "processed_features": None,
    "vif_results": None,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str) -> Tuple[pd.DataFrame, float]:
    """Weight of Evidence and Information Value for categorical features."""
    lst = []
    for val in df[feature].unique():
        good_cnt = df[(df[feature] == val) & (df[target] == 0)].shape[0]
        bad_cnt = df[(df[feature] == val) & (df[target] == 1)].shape[0]
        total_cnt = good_cnt + bad_cnt
        lst.append([feature, val, good_cnt, bad_cnt, total_cnt])

    data = pd.DataFrame(lst, columns=["Variable", "Value", "Good", "Bad", "Total"])
    total_good = data["Good"].sum()
    total_bad = data["Bad"].sum()

    data["Good%"] = (data["Good"] + 0.5) / (total_good + 0.5)
    data["Bad%"] = (data["Bad"] + 0.5) / (total_bad + 0.5)
    data["WoE"] = np.log(data["Good%"] / data["Bad%"])
    data["IV_Contribution"] = (data["Good%"] - data["Bad%"]) * data["WoE"]
    iv = data["IV_Contribution"].sum()
    return data, iv


def monotonic_woe_binning(df: pd.DataFrame, feature: str, target: str,
                          n_bins: int = 5) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Monotonic WoE binning for numerical features."""
    df_temp = df[[feature, target]].dropna().copy()
    df_temp["bin"], bins = pd.qcut(df_temp[feature], q=n_bins, duplicates="drop", retbins=True)

    total_good = (df_temp[target] == 0).sum()
    total_bad = (df_temp[target] == 1).sum()

    woe_bins = []
    for i, b in enumerate(sorted(df_temp["bin"].unique())):
        bin_data = df_temp[df_temp["bin"] == b]
        good_cnt = (bin_data[target] == 0).sum()
        bad_cnt = (bin_data[target] == 1).sum()
        total = good_cnt + bad_cnt
        good_pct = (good_cnt + 0.5) / total_good
        bad_pct = (bad_cnt + 0.5) / total_bad
        woe = np.log(good_pct / bad_pct) if bad_pct > 0 else 0
        woe_bins.append({
            "Bin": str(b),
            "Bin_Index": i,
            "Good": int(good_cnt),
            "Bad": int(bad_cnt),
            "Total": int(total),
            "Good%": round(good_pct, 6),
            "Bad%": round(bad_pct, 6),
            "WoE": round(woe, 6),
            "Default_Rate": round(bad_cnt / total, 6) if total > 0 else 0,
        })

    result_df = pd.DataFrame(woe_bins)
    result_df["IV_Contribution"] = (result_df["Good%"] - result_df["Bad%"]) * result_df["WoE"]
    iv = result_df["IV_Contribution"].sum()
    return bins, result_df, iv


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> Tuple[float, pd.DataFrame]:
    """Population Stability Index."""
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    if len(np.unique(expected)) < buckets:
        breakpoints = np.linspace(0, 100, len(np.unique(expected)) + 1)

    try:
        expected_percents = np.percentile(expected, breakpoints)
        actual_percents = np.percentile(actual, breakpoints)
    except Exception:
        return 0.0, pd.DataFrame()

    psi_values = []
    for i in range(len(breakpoints) - 1):
        exp_count = np.sum((expected >= expected_percents[i]) & (expected < expected_percents[i + 1]))
        act_count = np.sum((actual >= actual_percents[i]) & (actual < actual_percents[i + 1]))
        exp_pct = max(exp_count / len(expected), 0.0001)
        act_pct = max(act_count / len(actual), 0.0001)
        psi_value = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        psi_values.append({
            "Bucket": f"{breakpoints[i]:.1f}% - {breakpoints[i+1]:.1f}%",
            "Expected_Count": int(exp_count),
            "Actual_Count": int(act_count),
            "Expected%": round(exp_pct * 100, 4),
            "Actual%": round(act_pct * 100, 4),
            "PSI": round(psi_value, 6),
        })

    psi_df = pd.DataFrame(psi_values)
    return float(psi_df["PSI"].sum()), psi_df


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor."""
    vif_values = []
    for col in X.columns:
        y = X[col]
        X_other = X.drop(columns=[col])
        if X_other.empty:
            vif_values.append(1.0)
            continue
        m = LinearRegression()
        m.fit(X_other, y)
        r2 = m.score(X_other, y)
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_values.append(round(vif, 4))
    return pd.DataFrame({"Feature": X.columns, "VIF": vif_values})


def interpret_iv(iv: float) -> str:
    if iv < 0.02:
        return "Not Useful"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    elif iv < 0.5:
        return "Strong"
    else:
        return "Very Strong / Suspicious"


def interpret_psi(psi: float) -> Tuple[str, str]:
    if psi < 0.1:
        return "No Significant Change", "success"
    elif psi < 0.25:
        return "Moderate Change - Review Needed", "warning"
    else:
        return "Significant Change - Action Required", "danger"


def calculate_ks(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(max(tpr - fpr))


def df_to_records(df: pd.DataFrame) -> list:
    """Safe conversion of DataFrame to list of dicts (handles NaN)."""
    return df.replace({np.nan: None}).to_dict(orient="records")


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == "admin" and password == "admin":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid credentials. Try admin / admin."
    return render_template("login.html", error=error)


@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    summary = {
        "dataset_loaded": state["train_df"] is not None,
        "cleaned": state["cleaned_df"] is not None,
        "woe_done": state["iv_summary"] is not None,
        "model_trained": state["model"] is not None,
        "scorecard_done": state["train_scores"] is not None,
        "monitoring_done": state["monitoring_df"] is not None,
    }
    train_shape = state["train_df"].shape if state["train_df"] is not None else None
    cleaned_shape = state["cleaned_df"].shape if state["cleaned_df"] is not None else None
    metrics = state.get("model_metrics")
    return render_template(
        "dashboard.html",
        summary=summary,
        train_shape=train_shape,
        cleaned_shape=cleaned_shape,
        metrics=metrics,
        target_col=state["target_col"],
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ============================================================
# UPLOAD
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    message = None
    error = None
    preview = None
    dtype_info = None
    shape = None
    columns = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload":
            file = request.files.get("file")
            if not file or file.filename == "":
                error = "No file selected."
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                try:
                    df = pd.read_csv(filepath)
                    state["train_df"] = df
                    # Reset downstream state
                    for k in ["cleaned_df", "iv_summary", "binning_result", "feature_bins",
                              "woe_mappings", "selected_features", "model", "scaler",
                              "X_train", "X_test", "y_train", "y_test", "train_scores",
                              "test_scores", "scorecard_params", "monitoring_df",
                              "model_metrics", "processed_features", "vif_results"]:
                        state[k] = None
                    message = f"Dataset loaded successfully! Shape: {df.shape}"
                except Exception as e:
                    error = f"Error reading file: {str(e)}"

        elif action == "set_target":
            target_col = request.form.get("target_col")
            if target_col and state["train_df"] is not None:
                state["target_col"] = target_col
                message = f"Target variable set to: {target_col}"
            else:
                error = "Upload a dataset first."

    df = state["train_df"]
    if df is not None:
        preview = df_to_records(df.head(10))
        dtype_info = df_to_records(pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values,
            "Null %": (df.isnull().sum() / len(df) * 100).round(2).values,
        }))
        shape = df.shape
        columns = df.columns.tolist()

    return render_template(
        "upload.html",
        message=message,
        error=error,
        preview=preview,
        dtype_info=dtype_info,
        shape=shape,
        columns=columns,
        target_col=state["target_col"],
        num_features=df.select_dtypes(include=[np.number]).shape[1] if df is not None else 0,
        cat_features=df.select_dtypes(include=["object"]).shape[1] if df is not None else 0,
    )


# ============================================================
# CLEANING
# ============================================================

@app.route("/cleaning", methods=["GET", "POST"])
def cleaning():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    message = None
    error = None
    quality_report = None
    cleaned_preview = None
    cleaned_shape = None
    target_dist = None

    df = state["train_df"]
    if df is None:
        return render_template("cleaning.html", error="Please upload a training dataset first.",
                               columns=None, target_col=state["target_col"])

    if request.method == "POST":
        action = request.form.get("action")

        if action == "set_target":
            state["target_col"] = request.form.get("target_col")
            message = f"Target set to: {state['target_col']}"

        elif action == "clean":
            missing_strategy = request.form.get("missing_strategy", "mean_median")
            outlier_strategy = request.form.get("outlier_strategy", "iqr")
            target_col = state["target_col"]

            cleaned_df = df.copy()

            # Missing value treatment
            if missing_strategy == "drop":
                cleaned_df = cleaned_df.dropna()
            elif missing_strategy == "mean_median":
                for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                    if cleaned_df[col].isnull().sum() > 0:
                        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                for col in cleaned_df.select_dtypes(include=["object"]).columns:
                    if cleaned_df[col].isnull().sum() > 0:
                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            else:  # mode
                for col in cleaned_df.columns:
                    if cleaned_df[col].isnull().sum() > 0:
                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)

            # Duplicates
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            dups_removed = initial_rows - len(cleaned_df)

            # Outlier treatment
            if outlier_strategy == "iqr":
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    if target_col and col == target_col:
                        continue
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    cleaned_df[col] = cleaned_df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            elif outlier_strategy == "zscore":
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                z = np.abs(stats.zscore(cleaned_df[num_cols].fillna(0)))
                cleaned_df = cleaned_df[(z < 3).all(axis=1)]

            state["cleaned_df"] = cleaned_df
            rows_removed = len(df) - len(cleaned_df)
            message = (
                f"Cleaning complete! Final shape: {cleaned_df.shape}. "
                f"Rows removed: {rows_removed} ({rows_removed/len(df)*100:.2f}%). "
                f"Duplicates removed: {dups_removed}."
            )

    # Build quality report
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values
    }).query("`Missing Count` > 0")

    quality_report = {
        "missing": df_to_records(missing_df),
        "duplicates": int(df.duplicated().sum()),
        "total_rows": len(df),
    }

    target_col = state["target_col"]
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts().reset_index()
        vc.columns = ["Value", "Count"]
        target_dist = df_to_records(vc)

    if state["cleaned_df"] is not None:
        cleaned_preview = df_to_records(state["cleaned_df"].head(10))
        cleaned_shape = state["cleaned_df"].shape

    return render_template(
        "cleaning.html",
        message=message,
        error=error,
        quality_report=quality_report,
        columns=df.columns.tolist(),
        target_col=target_col,
        target_dist=target_dist,
        cleaned_preview=cleaned_preview,
        cleaned_shape=cleaned_shape,
    )


@app.route("/download/cleaned")
def download_cleaned():
    if state["cleaned_df"] is None:
        return redirect(url_for("cleaning"))
    buf = io.BytesIO()
    state["cleaned_df"].to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="cleaned_credit_data.csv")


# ============================================================
# WOE / IV
# ============================================================

@app.route("/woe", methods=["GET", "POST"])
def woe():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    message = None
    error = None
    iv_summary = None
    woe_detail = None
    selected_feature = None

    df = state["cleaned_df"]
    target_col = state["target_col"]

    if df is None:
        return render_template("woe.html", error="Please clean the data first.")
    if not target_col:
        return render_template("woe.html", error="Please set the target variable first.")

    if request.method == "POST":
        action = request.form.get("action")

        if action == "run_woe":
            n_bins = int(request.form.get("n_bins", 5))
            iv_threshold = float(request.form.get("iv_threshold", 0.02))

            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_features = df.select_dtypes(include=["object"]).columns.tolist()
            if target_col in numeric_features:
                numeric_features.remove(target_col)

            iv_results = []
            woe_results = {}
            feature_bins = {}
            woe_mappings = {}

            for feature in numeric_features:
                try:
                    bins, woe_df, iv = monotonic_woe_binning(df, feature, target_col, n_bins)
                    iv_results.append({
                        "Feature": feature,
                        "IV": round(iv, 6),
                        "Interpretation": interpret_iv(iv),
                        "Type": "Numerical",
                    })
                    woe_results[feature] = df_to_records(woe_df.round(6))
                    feature_bins[feature] = bins.tolist()
                    woe_mappings[feature] = {str(r["Bin"]): r["WoE"] for r in woe_results[feature]}
                except Exception as e:
                    pass

            for feature in cat_features:
                try:
                    woe_df, iv = calculate_woe_iv(df, feature, target_col)
                    iv_results.append({
                        "Feature": feature,
                        "IV": round(iv, 6),
                        "Interpretation": interpret_iv(iv),
                        "Type": "Categorical",
                    })
                    woe_results[feature] = df_to_records(woe_df.round(6))
                except Exception as e:
                    pass

            iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False)
            state["iv_summary"] = iv_df
            state["binning_result"] = woe_results
            state["feature_bins"] = feature_bins
            state["woe_mappings"] = woe_mappings

            # Auto-select features above threshold
            state["selected_features"] = iv_df[iv_df["IV"] >= iv_threshold]["Feature"].tolist()
            message = f"WoE/IV analysis complete! {len(iv_results)} features processed."

        elif action == "confirm_features":
            iv_threshold = float(request.form.get("iv_threshold", 0.02))
            if state["iv_summary"] is not None:
                state["selected_features"] = state["iv_summary"][
                    state["iv_summary"]["IV"] >= iv_threshold
                ]["Feature"].tolist()
                message = f"{len(state['selected_features'])} features selected for modeling."

        elif action == "view_detail":
            selected_feature = request.form.get("detail_feature")

    if state["iv_summary"] is not None:
        iv_summary = df_to_records(state["iv_summary"])

    if selected_feature and state["binning_result"] and selected_feature in state["binning_result"]:
        woe_detail = {"feature": selected_feature, "rows": state["binning_result"][selected_feature]}

    all_woe_features = list(state["binning_result"].keys()) if state["binning_result"] else []

    return render_template(
        "woe.html",
        message=message,
        error=error,
        iv_summary=iv_summary,
        woe_detail=woe_detail,
        all_woe_features=all_woe_features,
        selected_features=state["selected_features"],
        target_col=target_col,
    )


# ============================================================
# MODEL
# ============================================================

@app.route("/model", methods=["GET", "POST"])
def model():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    message = None
    error = None
    metrics = None
    coef_data = None
    cm_data = None
    report_data = None
    vif_data = None
    roc_data = None

    df = state["cleaned_df"]
    target_col = state["target_col"]
    iv_summary = state["iv_summary"]

    if df is None or iv_summary is None:
        return render_template("model.html", error="Please complete WoE/IV analysis first.")

    good_iv_features = iv_summary[iv_summary["IV"] >= 0.02]["Feature"].tolist()
    available_features = [
        f for f in good_iv_features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    if request.method == "POST":
        action = request.form.get("action")

        if action == "vif":
            selected = request.form.getlist("features")
            if not selected:
                error = "Select at least two features."
            else:
                try:
                    X_vif = df[selected].fillna(df[selected].median())
                    vif_df = calculate_vif(X_vif)
                    state["vif_results"] = df_to_records(vif_df)
                    message = "VIF calculated."
                except Exception as e:
                    error = str(e)

        elif action == "train":
            selected = request.form.getlist("features")
            test_size = int(request.form.get("test_size", 20)) / 100
            random_state = int(request.form.get("random_state", 42))
            class_weight = request.form.get("class_weight", "balanced")
            class_weight = class_weight if class_weight != "None" else None

            if not selected:
                error = "Select at least one feature."
            else:
                try:
                    X = df[selected].fillna(df[selected].median())
                    y = df[target_col]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )

                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_test_sc = scaler.transform(X_test)

                    clf = LogisticRegression(
                        class_weight=class_weight, max_iter=1000, random_state=random_state
                    )
                    clf.fit(X_train_sc, y_train)

                    state["model"] = clf
                    state["scaler"] = scaler
                    state["X_train"] = X_train
                    state["X_test"] = X_test
                    state["y_train"] = y_train
                    state["y_test"] = y_test
                    state["processed_features"] = selected

                    # Metrics
                    y_prob_train = clf.predict_proba(X_train_sc)[:, 1]
                    y_prob_test = clf.predict_proba(X_test_sc)[:, 1]
                    y_pred_test = clf.predict(X_test_sc)

                    train_auc = roc_auc_score(y_train, y_prob_train)
                    test_auc = roc_auc_score(y_test, y_prob_test)
                    ks_train = calculate_ks(y_train, y_prob_train)
                    ks_test = calculate_ks(y_test, y_prob_test)

                    state["model_metrics"] = {
                        "train_auc": round(train_auc, 4),
                        "test_auc": round(test_auc, 4),
                        "train_gini": round(2 * train_auc - 1, 4),
                        "test_gini": round(2 * test_auc - 1, 4),
                        "train_ks": round(ks_train, 4),
                        "test_ks": round(ks_test, 4),
                    }

                    # ROC curve data
                    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
                    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
                    state["roc_data"] = {
                        "fpr_train": fpr_train.tolist(),
                        "tpr_train": tpr_train.tolist(),
                        "fpr_test": fpr_test.tolist(),
                        "tpr_test": tpr_test.tolist(),
                        "train_auc": round(train_auc, 3),
                        "test_auc": round(test_auc, 3),
                    }

                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred_test)
                    state["cm_data"] = cm.tolist()

                    # Classification report
                    report = classification_report(y_test, y_pred_test, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().reset_index()
                    report_df.columns = ["Class"] + list(report_df.columns[1:])
                    state["report_data"] = df_to_records(report_df.round(4))

                    # Coefficients
                    coef_df = pd.DataFrame({
                        "Feature": selected,
                        "Coefficient": clf.coef_[0],
                    }).sort_values("Coefficient", key=abs, ascending=False)
                    state["coef_data"] = df_to_records(coef_df.round(6))

                    message = "Model trained successfully!"
                except Exception as e:
                    error = f"Training error: {str(e)}"

    metrics = state.get("model_metrics")
    coef_data = state.get("coef_data")
    cm_data = state.get("cm_data")
    report_data = state.get("report_data")
    vif_data = state.get("vif_results")
    roc_data = state.get("roc_data")

    return render_template(
        "model.html",
        message=message,
        error=error,
        available_features=available_features,
        selected_features=state["processed_features"],
        metrics=metrics,
        coef_data=coef_data,
        cm_data=cm_data,
        report_data=report_data,
        vif_data=vif_data,
        roc_data=roc_data,
    )


# ============================================================
# SCORECARD
# ============================================================

@app.route("/scorecard", methods=["GET", "POST"])
def scorecard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    message = None
    error = None
    score_stats = None
    train_score_hist = None
    test_score_hist = None
    segment_summary = None
    sample_scores = None

    if state["model"] is None:
        return render_template("scorecard.html", error="Please train the model first.")

    if request.method == "POST":
        try:
            base_score = float(request.form.get("base_score", 600))
            base_odds = float(request.form.get("base_odds", 20))
            pdo = float(request.form.get("pdo", 20))
            low_threshold = float(request.form.get("low_threshold", 700))
            high_threshold = float(request.form.get("high_threshold", 550))

            factor = pdo / np.log(2)
            offset = base_score - factor * np.log(base_odds)

            clf = state["model"]
            scaler = state["scaler"]
            X_train = state["X_train"]
            X_test = state["X_test"]
            y_test = state["y_test"]

            X_train_sc = scaler.transform(X_train)
            X_test_sc = scaler.transform(X_test)

            train_pd = clf.predict_proba(X_train_sc)[:, 1]
            test_pd = clf.predict_proba(X_test_sc)[:, 1]

            train_odds = (1 - train_pd) / np.maximum(train_pd, 1e-10)
            test_odds = (1 - test_pd) / np.maximum(test_pd, 1e-10)

            train_scores = offset + factor * np.log(np.maximum(train_odds, 1e-10))
            test_scores = offset + factor * np.log(np.maximum(test_odds, 1e-10))

            state["train_scores"] = train_scores
            state["test_scores"] = test_scores
            state["scorecard_params"] = {
                "base_score": base_score,
                "base_odds": base_odds,
                "pdo": pdo,
                "factor": round(factor, 4),
                "offset": round(offset, 4),
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
            }

            # Score statistics
            score_stats = {
                "train": {
                    "min": round(float(train_scores.min()), 1),
                    "max": round(float(train_scores.max()), 1),
                    "mean": round(float(train_scores.mean()), 1),
                    "std": round(float(train_scores.std()), 1),
                },
                "test": {
                    "min": round(float(test_scores.min()), 1),
                    "max": round(float(test_scores.max()), 1),
                    "mean": round(float(test_scores.mean()), 1),
                    "std": round(float(test_scores.std()), 1),
                },
            }

            # Histogram bins for chart
            def make_hist(scores, n=30):
                counts, edges = np.histogram(scores, bins=n)
                labels = [f"{edges[i]:.0f}" for i in range(len(edges) - 1)]
                return {"labels": labels, "counts": counts.tolist()}

            train_score_hist = make_hist(train_scores)
            test_score_hist = make_hist(test_scores)

            # Segmentation
            def segment(s):
                if s >= low_threshold:
                    return "Low Risk"
                elif s >= high_threshold:
                    return "Medium Risk"
                return "High Risk"

            train_segs = pd.Series(train_scores).apply(segment).value_counts()
            test_segs = pd.Series(test_scores).apply(segment).value_counts()

            segment_summary = []
            for seg, decision in [("Low Risk", "Approve"), ("Medium Risk", "Manual Review"), ("High Risk", "Reject")]:
                tc = int(train_segs.get(seg, 0))
                ec = int(test_segs.get(seg, 0))
                segment_summary.append({
                    "Segment": seg,
                    "Decision": decision,
                    "Score Range": f">= {low_threshold:.0f}" if seg == "Low Risk"
                                  else f"{high_threshold:.0f} - {low_threshold:.0f}" if seg == "Medium Risk"
                                  else f"< {high_threshold:.0f}",
                    "Train Count": tc,
                    "Train %": f"{tc/len(train_scores)*100:.1f}%",
                    "Test Count": ec,
                    "Test %": f"{ec/len(test_scores)*100:.1f}%",
                })

            # Sample scores table
            sample_idx = np.random.choice(len(test_scores), min(20, len(test_scores)), replace=False)
            sample_records = []
            for i in sample_idx:
                sample_records.append({
                    "Index": int(i),
                    "Score": round(float(test_scores[i]), 1),
                    "PD": round(float(test_pd[i]), 4),
                    "Segment": segment(test_scores[i]),
                    "Actual": int(list(y_test)[i]),
                })
            sample_scores = sorted(sample_records, key=lambda x: x["Score"], reverse=True)

            message = "Scores calculated successfully!"
        except Exception as e:
            error = f"Scorecard error: {str(e)}"

    params = state.get("scorecard_params")
    if state["train_scores"] is not None and score_stats is None:
        # Re-populate for GET after already calculated
        ts = state["train_scores"]
        es = state["test_scores"]
        score_stats = {
            "train": {"min": round(float(ts.min()), 1), "max": round(float(ts.max()), 1),
                      "mean": round(float(ts.mean()), 1), "std": round(float(ts.std()), 1)},
            "test": {"min": round(float(es.min()), 1), "max": round(float(es.max()), 1),
                     "mean": round(float(es.mean()), 1), "std": round(float(es.std()), 1)},
        }

    return render_template(
        "scorecard.html",
        message=message,
        error=error,
        params=params,
        score_stats=score_stats,
        train_score_hist=train_score_hist,
        test_score_hist=test_score_hist,
        segment_summary=segment_summary,
        sample_scores=sample_scores,
    )


# ============================================================
# MONITOR
# ============================================================

@app.route("/monitor", methods=["GET", "POST"])
def monitor():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    message = None
    error = None
    psi_summary = None
    score_psi = None
    score_psi_status = None
    recommendations = None
    monitor_score_hist = None
    train_score_hist = None
    feature_dist = None
    selected_feature = None

    if state["model"] is None or state["train_scores"] is None:
        return render_template("monitor.html",
                               error="Please complete the full pipeline (through Scorecard) first.")

    train_df = state["cleaned_df"]
    features = state["processed_features"]

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload_monitor":
            file = request.files.get("monitor_file")
            if not file or file.filename == "":
                error = "No file selected."
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], "monitor_" + filename)
                file.save(filepath)
                try:
                    mdf = pd.read_csv(filepath)
                    # Keep only columns that exist in both
                    common_cols = [c for c in features if c in mdf.columns]
                    if not common_cols:
                        error = "Monitoring dataset has no matching feature columns."
                    else:
                        state["monitoring_df"] = mdf
                        message = f"Monitoring dataset loaded. Shape: {mdf.shape}"
                except Exception as e:
                    error = f"Error reading monitoring file: {str(e)}"

        elif action == "run_monitor":
            mdf = state["monitoring_df"]
            if mdf is None:
                error = "Upload a monitoring dataset first."
            else:
                try:
                    clf = state["model"]
                    scaler = state["scaler"]
                    train_scores = state["train_scores"]
                    params = state["scorecard_params"]

                    common_features = [f for f in features if f in mdf.columns]

                    X_monitor = mdf[common_features].fillna(mdf[common_features].median())
                    X_monitor_sc = scaler.transform(X_monitor[features] if all(f in X_monitor.columns for f in features)
                                                    else X_monitor)
                    monitor_pd = clf.predict_proba(X_monitor_sc)[:, 1]
                    monitor_odds = (1 - monitor_pd) / np.maximum(monitor_pd, 1e-10)
                    monitor_scores = params["offset"] + params["factor"] * np.log(np.maximum(monitor_odds, 1e-10))

                    # PSI per feature
                    psi_rows = []
                    for feat in common_features:
                        try:
                            psi_val, _ = calculate_psi(
                                train_df[feat].dropna().values,
                                mdf[feat].dropna().values
                            )
                            status, color = interpret_psi(psi_val)
                            psi_rows.append({
                                "Feature": feat,
                                "PSI": round(psi_val, 4),
                                "Status": status,
                                "Color": color,
                            })
                        except Exception:
                            pass

                    psi_summary = sorted(psi_rows, key=lambda x: x["PSI"], reverse=True)

                    # Score PSI
                    score_psi_val, _ = calculate_psi(train_scores, monitor_scores)
                    score_psi = round(float(score_psi_val), 4)
                    score_psi_status, _ = interpret_psi(score_psi_val)

                    # Score histograms
                    def make_hist(scores, n=30):
                        counts, edges = np.histogram(scores, bins=n)
                        return {
                            "labels": [f"{edges[i]:.0f}" for i in range(len(edges) - 1)],
                            "counts": counts.tolist(),
                        }

                    train_score_hist = make_hist(train_scores)
                    monitor_score_hist = make_hist(monitor_scores)

                    # Recommendations
                    recs = []
                    major_drift = [r for r in psi_summary if r["PSI"] > 0.25]
                    moderate_drift = [r for r in psi_summary if 0.1 <= r["PSI"] <= 0.25]

                    if major_drift:
                        recs.append({
                            "Priority": "HIGH",
                            "Issue": f"{len(major_drift)} feature(s) with major drift (PSI > 0.25)",
                            "Recommendation": "Immediate model retraining recommended. Features: "
                                              + ", ".join(r["Feature"] for r in major_drift),
                        })
                    if moderate_drift:
                        recs.append({
                            "Priority": "MEDIUM",
                            "Issue": f"{len(moderate_drift)} feature(s) with moderate drift",
                            "Recommendation": "Monitor closely. Consider model recalibration. Features: "
                                              + ", ".join(r["Feature"] for r in moderate_drift),
                        })
                    if score_psi_val > 0.25:
                        recs.append({
                            "Priority": "HIGH",
                            "Issue": f"Score PSI ({score_psi:.3f}) indicates major drift",
                            "Recommendation": "Review scorecard and consider redevelopment",
                        })
                    if not recs:
                        recs.append({
                            "Priority": "LOW",
                            "Issue": "No significant drift detected",
                            "Recommendation": "Continue regular monitoring schedule",
                        })

                    recommendations = recs
                    message = "Monitoring analysis complete!"

                    # Store for feature dist view
                    state["monitor_scores"] = monitor_scores
                    state["psi_summary"] = psi_summary

                except Exception as e:
                    error = f"Monitoring error: {str(e)}"

        elif action == "feature_dist":
            selected_feature = request.form.get("dist_feature")
            mdf = state["monitoring_df"]
            if mdf is not None and selected_feature:
                def make_hist_vals(series, n=20):
                    vals = series.dropna().values
                    counts, edges = np.histogram(vals, bins=n)
                    return {
                        "labels": [f"{edges[i]:.2f}" for i in range(len(edges) - 1)],
                        "counts": counts.tolist(),
                    }

                feature_dist = {
                    "feature": selected_feature,
                    "train": make_hist_vals(train_df[selected_feature]) if selected_feature in train_df.columns else None,
                    "monitor": make_hist_vals(mdf[selected_feature]) if selected_feature in mdf.columns else None,
                }

    psi_summary = psi_summary or state.get("psi_summary")
    monitor_loaded = state["monitoring_df"] is not None
    monitor_features = [f for f in features if state["monitoring_df"] is not None
                        and f in state["monitoring_df"].columns] if state["monitoring_df"] is not None else []

    return render_template(
        "monitor.html",
        message=message,
        error=error,
        monitor_loaded=monitor_loaded,
        monitor_shape=state["monitoring_df"].shape if state["monitoring_df"] is not None else None,
        psi_summary=psi_summary,
        score_psi=score_psi,
        score_psi_status=score_psi_status,
        recommendations=recommendations,
        train_score_hist=train_score_hist,
        monitor_score_hist=monitor_score_hist,
        feature_dist=feature_dist,
        selected_feature=selected_feature,
        monitor_features=monitor_features,
    )


@app.route("/download/monitor_report")
def download_monitor_report():
    if state["monitoring_df"] is None or state["psi_summary"] is None:
        return redirect(url_for("monitor"))
    psi_df = pd.DataFrame(state["psi_summary"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        psi_df.to_excel(writer, sheet_name="PSI Summary", index=False)
    buf.seek(0)
    return send_file(buf,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True,
                     download_name="credit_risk_monitoring_report.xlsx")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
