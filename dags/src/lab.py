"""
Customer segmentation pipeline for a food wholesale distributor.

Modification of MLOps Airflow Lab 1. The original K-Means + elbow-method pipeline
is applied to the UCI Wholesale Customers dataset (annual spending of 440 hotels,
restaurants, cafes and shops in Portugal) and extended with:
  - a data validation step that stops the run on bad input,
  - silhouette scores as a second measure of cluster quality,
  - a quality gate that only publishes segments when they are well separated,
  - business-friendly segment names and an HTML report with charts.
"""

import base64
import io
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside Docker
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kneed import KneeLocator  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DAGS_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = DAGS_DIR / "data" / "wholesale_customers.csv"
MODEL_DIR = DAGS_DIR / "model"
OUTPUT_DIR = DAGS_DIR.parent / "working_data"  # /opt/airflow/working_data in Docker

SPENDING_COLUMNS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
REQUIRED_COLUMNS = ["Channel", "Region"] + SPENDING_COLUMNS
CHANNEL_NAMES = {1: "Horeca", 2: "Retail"}  # Horeca = hotel/restaurant/cafe
CATEGORY_LABELS = {
    "Fresh": "Fresh produce", "Milk": "Dairy", "Grocery": "Grocery",
    "Frozen": "Frozen", "Detergents_Paper": "Cleaning & paper", "Delicassen": "Deli",
}
RANDOM_STATE = 42


def _encode(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _decode(text):
    return pickle.loads(base64.b64decode(text))


def _features(df):
    """Log-transform skewed spending, then standardize."""
    return StandardScaler().fit_transform(np.log1p(df[SPENDING_COLUMNS]))


# ----------------------------------------------------------------- tasks

def load_data():
    """Read the customer CSV and return it serialized for XCom."""
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} customers from {DATA_FILE.name}")
    return _encode(df)


def validate_data(data, min_rows=100):
    """Data contract: fail the run early if the input is not what the model expects."""
    df = _decode(data)
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    else:
        if len(df) < min_rows:
            problems.append(f"only {len(df)} rows (need at least {min_rows})")
        if df[REQUIRED_COLUMNS].isna().any().any():
            problems.append("null values found")
        if (df[SPENDING_COLUMNS] < 0).any().any():
            problems.append("negative spending values found")
        if not set(df["Channel"].unique()) <= set(CHANNEL_NAMES):
            problems.append("unknown Channel codes")
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    print(f"Data validation passed: {len(df)} rows, {len(df.columns)} columns, "
          "no nulls or negative values")
    return {"rows": int(len(df)), "columns": int(len(df.columns))}


def data_preprocessing(data):
    """Return log-transformed, standardized spending features."""
    X = _features(_decode(data))
    print(f"Preprocessed {X.shape[0]} customers x {X.shape[1]} spending features")
    return _encode(X)


def build_save_model(data, filename, max_k=10):
    """Fit K-Means for k = 1..max_k, save all models, return SSE and silhouette per k."""
    X = _decode(data)
    sweep = {"k": [], "sse": [], "silhouette": []}
    models = {}
    for k in range(1, int(max_k) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        models[k] = km
        sil = float(silhouette_score(X, km.labels_)) if k > 1 else None
        sweep["k"].append(k)
        sweep["sse"].append(float(km.inertia_))
        sweep["silhouette"].append(sil)
        shown = "-" if sil is None else f"{sil:.3f}"
        print(f"k={k:2d}  SSE={km.inertia_:8.1f}  silhouette={shown}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / filename, "wb") as f:
        pickle.dump(models, f)
    return sweep


def load_model_elbow(filename, sweep):
    """Load the saved models and choose the number of segments with the elbow method."""
    with open(MODEL_DIR / filename, "rb") as f:
        models = pickle.load(f)
    kl = KneeLocator(sweep["k"], sweep["sse"], curve="convex", direction="decreasing")
    k = int(kl.elbow) if kl.elbow else 3
    if k not in models:
        raise ValueError(f"No saved model for k={k}")
    print(f"Elbow method chose k = {k}")
    return k


def quality_gate(sweep, k, min_silhouette=0.15):
    """Branch: publish segments only if the chosen clustering is well separated."""
    k, min_silhouette = int(k), float(min_silhouette)
    score = sweep["silhouette"][sweep["k"].index(k)]
    if score is None:
        print("k = 1 has no silhouette score -> cannot publish a single segment")
        return "flag_low_quality_task"
    print(f"Silhouette at k={k}: {score:.3f} (minimum {min_silhouette})")
    if score >= min_silhouette:
        print("Quality gate PASSED -> publishing segments")
        return "segment_customers_task"
    print("Quality gate FAILED -> segments will not be published")
    return "flag_low_quality_task"


def name_segments(profile, overall_mean):
    """Name each segment by the two categories it buys most, relative to the average customer."""
    names = {}
    for seg, row in profile.iterrows():
        lift = (row[SPENDING_COLUMNS].astype(float) / overall_mean).sort_values(ascending=False)
        top = [CATEGORY_LABELS[c] for c in lift.index[:2]]
        names[seg] = f"{row['Main_Channel']}: {top[0]} + {top[1]}"
    return names


def segment_customers(filename, k, sweep):
    """Assign customers to segments, write CSVs and an HTML report for the sales team."""
    k = int(k)
    df = pd.read_csv(DATA_FILE)
    with open(MODEL_DIR / filename, "rb") as f:
        model = pickle.load(f)[k]
    df["Segment"] = model.predict(_features(df))

    profile = df.groupby("Segment")[SPENDING_COLUMNS].mean().round(0)
    profile["Customers"] = df["Segment"].value_counts().sort_index()
    profile["Share_pct"] = (100 * profile["Customers"] / len(df)).round(1)
    profile["Main_Channel"] = (
        df.groupby("Segment")["Channel"].agg(lambda s: s.mode()[0]).map(CHANNEL_NAMES)
    )
    profile["Segment_Name"] = pd.Series(name_segments(profile, df[SPENDING_COLUMNS].mean()))
    df["Segment_Name"] = df["Segment"].map(profile["Segment_Name"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "segment_profiles.csv")
    _write_html_report(profile, sweep, k)

    print(profile[["Segment_Name", "Customers", "Share_pct"] + SPENDING_COLUMNS].to_string())
    return {"segments": k, "customers": int(len(df)),
            "report": "working_data/segment_report.html"}


# ----------------------------------------------------------------- report

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_html_report(profile, sweep, k):
    # Chart 1: elbow curve with silhouette on a second axis
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax1.plot(sweep["k"], sweep["sse"], "o-", color="#2563eb", label="SSE (inertia)")
    ax1.axvline(k, color="#dc2626", linestyle="--", label=f"chosen k = {k}")
    ax1.set_xlabel("Number of segments (k)")
    ax1.set_ylabel("SSE")
    ax2 = ax1.twinx()
    ax2.plot(sweep["k"][1:], sweep["silhouette"][1:], "s--", color="#16a34a", label="Silhouette")
    ax2.set_ylabel("Silhouette")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=8)
    ax1.set_title("Choosing the number of segments")
    elbow_png = _fig_to_base64(fig)

    # Chart 2: average spend per category for each segment
    fig, ax = plt.subplots(figsize=(8, 3.8))
    plot_df = profile.set_index("Segment_Name")[SPENDING_COLUMNS].rename(columns=CATEGORY_LABELS)
    plot_df.T.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Average annual spend (m.u.)")
    ax.set_title("What each segment buys")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    spend_png = _fig_to_base64(fig)

    rows = ""
    for _, r in profile.iterrows():
        spend_cells = "".join(f"<td>{int(r[c]):,}</td>" for c in SPENDING_COLUMNS)
        rows += (f"<tr><td><b>{r['Segment_Name']}</b></td><td>{int(r['Customers'])}</td>"
                 f"<td>{r['Share_pct']}%</td>{spend_cells}</tr>")
    headers = "".join(f"<th>{CATEGORY_LABELS[c]}</th>" for c in SPENDING_COLUMNS)
    sil = sweep["silhouette"][sweep["k"].index(k)]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Wholesale Customer Segments</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 960px; margin: 32px auto;
        padding: 0 16px; color: #1f2937; }}
 h1 {{ margin-bottom: 4px; }} .sub {{ color: #6b7280; margin-top: 0; }}
 .cards {{ display: flex; gap: 12px; margin: 20px 0; }}
 .card {{ flex: 1; background: #f3f4f6; border-radius: 10px; padding: 14px; }}
 .card b {{ display: block; font-size: 26px; color: #2563eb; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }} th {{ background: #f9fafb; }}
 img {{ max-width: 100%; margin: 16px 0; }}
</style></head><body>
<h1>Wholesale Customer Segments</h1>
<p class="sub">Generated by the Airflow DAG <code>Wholesale_Customer_Segmentation</code>
 on {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="cards">
 <div class="card"><b>{int(profile['Customers'].sum())}</b>customers</div>
 <div class="card"><b>{k}</b>segments (elbow method)</div>
 <div class="card"><b>{sil:.2f}</b>silhouette score</div>
</div>
<h2>Segment profiles</h2>
<table><tr><th>Segment</th><th>Customers</th><th>Share</th>{headers}</tr>{rows}</table>
<p class="sub">Spending columns show average annual spend per customer (monetary units).</p>
<img src="data:image/png;base64,{spend_png}" alt="Spend by segment">
<img src="data:image/png;base64,{elbow_png}" alt="Elbow curve and silhouette scores">
</body></html>"""
    (OUTPUT_DIR / "segment_report.html").write_text(html, encoding="utf-8")
