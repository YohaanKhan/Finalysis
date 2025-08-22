from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
import json
import pandas as pd
from xgboost import XGBRegressor
from ingestion import run_pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

app = Flask(__name__, template_folder=str(HERE / "templates"))
app.secret_key = "dev-secret-key"

DEFAULT_MODEL_NAME = "credit_rating_xgboost_final.json"
ALTERNATE_MODEL = "credit_rating_xgboost_baseline.json"
WEIGHT_THRESHOLD_PERCENT = 2.0

METRIC_MAP = {
    "Current Ratio": ["current ratio", "current_ratio", "currentratio", "currentratio"],
    "Debt-to-Equity Ratio": ["debt to equity", "debt_to_equity", "debt_equity", "debt-to-equity"],
    "Gross Profit Margin": ["gross profit margin", "gross_profit_margin", "gpm"],
    "Operating / EBIT Margin": ["operating margin", "ebit margin", "operating_margin", "ebit_margin"],
    "EBITDA Margin": ["ebitda margin", "ebitda_margin", "ebitda"],
    "Pre-Tax Profit Margin": ["pre-tax profit margin", "pre_tax_profit_margin", "pretax_margin"],
    "Net Profit Margin": ["net profit margin", "net_profit_margin", "npm"],
    "Asset Turnover": ["asset turnover", "asset_turnover", "assetturnover"],
    "Return on Equity (ROE)": ["return on equity", "roe", "return_on_equity"],
    "Return on Assets (ROA)": ["return on assets", "roa", "return_on_assets"],
    "Operating Cash Flow per Share": ["operating cash flow per share", "op_cf_per_share", "ocfps"],
    "Free Cash Flow (FCF)": ["free cash flow", "fcf", "free_cash_flow"]
}

def load_model(path: Path) -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(str(path))
    return m

def feature_weights_percent(model, feature_cols):
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    weights = {col: float(raw.get(f"f{i}", 0.0)) for i, col in enumerate(feature_cols)}
    total = sum(weights.values())
    if total > 0:
        return {k: (v / total) * 100.0 for k, v in weights.items()}
    fi = model.feature_importances_
    tot = float(fi.sum()) if fi.sum() != 0 else 1.0
    return {col: float((fi[i] / tot) * 100.0) for i, col in enumerate(feature_cols)}

def lookup_metric_value(row, name_variants):
    for variant in name_variants:
        for col in row.index:
            if col.lower().strip() == variant.lower().strip():
                return row[col]
    # try fuzzy by contains
    for variant in name_variants:
        for col in row.index:
            if variant.lower().strip() in col.lower().strip():
                return row[col]
    return None

def evaluate_threshold(display_name, value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    try:
        v = float(value)
    except Exception:
        return "unknown"
    n = display_name.lower()
    if "current ratio" in n:
        if v > 1.5: return "good"
        if 1.0 <= v <= 1.5: return "medium"
        return "bad"
    if "debt" in n and "equity" in n:
        if v < 1.0: return "good"
        if 1.0 <= v <= 2.0: return "medium"
        return "bad"
    if "gross profit" in n:
        if v >= 30: return "good"
        if 20 <= v < 30: return "medium"
        return "bad"
    if "operating" in n or "ebit" in n:
        if v >= 15: return "good"
        if 5 <= v < 15: return "medium"
        return "bad"
    if "ebitda" in n:
        if v >= 20: return "good"
        if 10 <= v < 20: return "medium"
        return "bad"
    if "pre-tax" in n or "pre tax" in n or "pre_tax" in n:
        if v >= 20: return "good"
        if 10 <= v < 20: return "medium"
        return "bad"
    if "net profit" in n:
        if v >= 20: return "good"
        if 10 <= v < 20: return "medium"
        if v < 5: return "bad"
        return "medium"
    if "asset turnover" in n:
        if v > 1.0: return "good"
        if 0.5 <= v <= 1.0: return "medium"
        return "bad"
    if "return on equity" in n or "roe" in n:
        if v >= 15: return "good"
        if 10 <= v < 15: return "medium"
        return "bad"
    if "return on assets" in n or "roa" in n:
        if v > 20: return "good"
        if 5 <= v <= 20: return "medium"
        return "bad"
    if "operating cash" in n and "per" in n:
        if v > 0: return "good"
        if v == 0: return "medium"
        return "bad"
    if "free cash flow" in n or "fcf" in n:
        if v > 0: return "good"
        if v == 0: return "medium"
        return "bad"
    return "unknown"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    ticker = request.form.get("ticker", "").strip().upper()
    if not ticker:
        flash("Enter ticker", "error")
        return redirect(url_for("index"))

    run_pipeline(ticker)

    structured = DATA_DIR / f"{ticker}_structured.csv"
    df = pd.read_csv(structured)
    X = df.drop(columns=["Rating Date"], errors="ignore")
    X_latest = X.iloc[[-1]].reset_index(drop=True)
    row = X_latest.loc[0]
    feature_cols = X_latest.columns.tolist()

    model_path = MODELS_DIR / DEFAULT_MODEL_NAME
    if not model_path.exists():
        model_path = MODELS_DIR / ALTERNATE_MODEL
    model = load_model(model_path)
    y = model.predict(X_latest)[0]
    rating = float(y)

    weights = feature_weights_percent(model, feature_cols)
    features = []
    for k, v in weights.items():
        features.append({"name": k, "weight": float(v), "display_percent": round(float(v), 3)})

    features = sorted(features, key=lambda x: x["weight"], reverse=True)
    total_percent = sum(f["display_percent"] for f in features)
    if total_percent == 0 and len(features) > 0:
        # fallback even distribution
        p = 100.0 / len(features)
        for f in features:
            f["display_percent"] = round(p, 3)

    # color and shade
    for f in features:
        w = f["display_percent"]
        f["status"] = "good" if w >= WEIGHT_THRESHOLD_PERCENT else "bad"
        # shade variation by weight: use HSL
        hue = 140 if f["status"] == "good" else 6
        light = max(30, min(70, int(70 - (w / 100.0) * 40)))
        f["color"] = f"hsl({hue} {90}% {light}%)"

    sentiment_file = DATA_DIR / f"{ticker}_sentiment_data.json"
    headlines = []
    avg_sentiment = None
    if sentiment_file.exists():
        with open(sentiment_file, "r", encoding="utf-8") as f:
            sdata = json.load(f)
            avg_sentiment = sdata.get("average_sentiment")
            headlines = sdata.get("headlines", [])[:20]

    # metrics to show: metrics defined in METRIC_MAP (thresholded) first
    metrics = []
    for display_name, variants in METRIC_MAP.items():
        val = lookup_metric_value(row, variants)
        status = evaluate_threshold(display_name, val)
        metrics.append({"display": display_name, "value": None if val is None else val, "status": status})

    # append other numeric columns until we reach 22 metrics
    if len(metrics) < 22:
        added = {m["display"] for m in metrics}
        for col in X_latest.columns:
            if col in added:
                continue
            val = row[col]
            if isinstance(val, (int, float)):
                metrics.append({"display": col, "value": val, "status": "unknown"})
            if len(metrics) >= 22:
                break

    return render_template("dashboard.html",
                           ticker=ticker,
                           rating=round(rating, 2),
                           features=features,
                           headlines=headlines,
                           avg_sentiment=avg_sentiment,
                           metrics=metrics,
                           model_file=model_path.name)

if __name__ == "__main__":
    print("ROOT:", ROOT)
    app.run(host="0.0.0.0", port=5000, debug=True)
