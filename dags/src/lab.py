"""
Customer segmentation pipeline for a food wholesale distributor.

Modification of MLOps Airflow Lab 1. The original K-Means + elbow-method pipeline
is applied to the UCI Wholesale Customers dataset (annual spending of 440 hotels,
restaurants, cafes and shops in Portugal) and extended with:
  - a data validation step that stops the run on bad input,
  - silhouette scores as a second measure of cluster quality,
  - a quality gate that only publishes segments when they are well separated,
  - business-friendly segment names and an HTML report with charts,
  - revenue per segment, growth opportunity, and a list of customers to target.
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
REGION_NAMES = {1: "Lisbon", 2: "Oporto", 3: "Other"}
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


def signature_category(row, overall_mean):
    """The category a segment buys most, relative to the average customer."""
    lift = (row[SPENDING_COLUMNS].astype(float) / overall_mean).sort_values(ascending=False)
    return lift.index[0], lift.index[1]


def name_segments(profile, overall_mean):
    """Name each segment by the two categories it buys most, relative to the average customer."""
    names = {}
    for seg, row in profile.iterrows():
        first, second = signature_category(row, overall_mean)
        names[seg] = f"{row['Main_Channel']}: {CATEGORY_LABELS[first]} + {CATEGORY_LABELS[second]}"
    return names


def find_promotion_targets(df, signature, main_channel, top_n=10):
    """
    Share-of-wallet gaps: a customer who spends less than similar customers on their
    segment's signature category probably buys it from a competitor.

    "Similar" means same segment AND same channel as the segment's main channel, so
    restaurants are only compared with restaurants and shops with shops (a restaurant
    in a retail-like segment is not a real lead for retail products).
    Returns the customers with the biggest gap and the total gap per segment.
    """
    rows = []
    for seg, category in signature.items():
        members = df[(df["Segment"] == seg) & (df["Channel"] == main_channel[seg])]
        median = members[category].median()
        gap = (median - members[category]).clip(lower=0)
        rows.append(pd.DataFrame({
            "Customer_ID": members["Customer_ID"],
            "Segment": seg,
            "Segment_Name": members["Segment_Name"],
            "Channel": members["Channel"].map(CHANNEL_NAMES),
            "Region": members["Region"].map(REGION_NAMES),
            "Category": CATEGORY_LABELS[category],
            "Current_Spend": members[category],
            "Peer_Median": median,
            "Gap": gap.round(0),
        }))
    gaps = pd.concat(rows)
    opportunity = gaps.groupby("Segment").agg(
        Opportunity=("Gap", "sum"), Customers_Below_Median=("Gap", lambda g: int((g > 0).sum())))
    targets = (gaps[gaps["Gap"] > 0].sort_values("Gap", ascending=False)
               .groupby("Segment").head(top_n).reset_index(drop=True))
    return targets, opportunity


def segment_customers(filename, k, sweep, top_n=10):
    """Assign customers to segments, size each segment's revenue and growth opportunity,
    and write CSVs plus an HTML report for the sales team."""
    k = int(k)
    df = pd.read_csv(DATA_FILE)
    df.insert(0, "Customer_ID", range(1, len(df) + 1))  # row number in the source file
    with open(MODEL_DIR / filename, "rb") as f:
        model = pickle.load(f)[k]
    df["Segment"] = model.predict(_features(df))
    df["Annual_Spend"] = df[SPENDING_COLUMNS].sum(axis=1)
    overall_mean = df[SPENDING_COLUMNS].mean()

    profile = df.groupby("Segment")[SPENDING_COLUMNS].mean().round(0)
    profile["Customers"] = df["Segment"].value_counts().sort_index()
    profile["Share_pct"] = (100 * profile["Customers"] / len(df)).round(1)
    main_channel = df.groupby("Segment")["Channel"].agg(lambda s: s.mode()[0])
    profile["Main_Channel"] = main_channel.map(CHANNEL_NAMES)
    profile["Segment_Name"] = pd.Series(name_segments(profile, overall_mean))
    df["Segment_Name"] = df["Segment"].map(profile["Segment_Name"])

    # Business value of each segment
    revenue = df.groupby("Segment")["Annual_Spend"]
    profile["Revenue"] = revenue.sum()
    profile["Revenue_pct"] = (100 * profile["Revenue"] / df["Annual_Spend"].sum()).round(1)
    profile["Avg_Spend"] = revenue.mean().round(0)

    # Growth opportunity in each segment's signature category
    signature = {seg: signature_category(row, overall_mean)[0] for seg, row in profile.iterrows()}
    profile["Signature_Category"] = pd.Series(signature).map(CATEGORY_LABELS)
    targets, opportunity = find_promotion_targets(df, signature, main_channel.to_dict(), int(top_n))
    profile = profile.join(opportunity)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "segment_profiles.csv")
    targets.to_csv(OUTPUT_DIR / "promotion_targets.csv", index=False)
    _write_html_report(profile, targets, sweep, k)

    print(profile[["Segment_Name", "Customers", "Revenue_pct", "Avg_Spend",
                   "Signature_Category", "Opportunity"]].to_string())
    print(f"\nTop promotion targets written to promotion_targets.csv ({len(targets)} customers)")
    return {"segments": k, "customers": int(len(df)),
            "opportunity": int(profile["Opportunity"].sum()),
            "report": "working_data/segment_report.html"}


# ----------------------------------------------------------------- report

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _money(x):
    return f"{int(round(x)):,}"


def _write_html_report(profile, targets, sweep, k):
    colors = plt.cm.tab10.colors

    # Chart 1: share of customers vs share of revenue
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.6 * len(profile)))
    y = np.arange(len(profile))
    bars_c = ax.barh(y - 0.2, profile["Share_pct"], height=0.4, color="#93c5fd", label="% of customers")
    bars_r = ax.barh(y + 0.2, profile["Revenue_pct"], height=0.4, color="#2563eb", label="% of revenue")
    for bars in (bars_c, bars_r):
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=3)
    ax.set_yticks(y, profile["Segment_Name"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(profile["Share_pct"].max(), profile["Revenue_pct"].max()) * 1.15)
    ax.set_xlabel("Percent")
    ax.set_title("Who the customers are vs. where the money comes from")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    share_png = _fig_to_base64(fig)

    # Chart 2: average spend per category for each segment
    fig, ax = plt.subplots(figsize=(8, 3.8))
    plot_df = profile.set_index("Segment_Name")[SPENDING_COLUMNS].rename(columns=CATEGORY_LABELS)
    plot_df.T.plot(kind="bar", ax=ax, width=0.8, color=colors[:len(profile)])
    ax.set_ylabel("Average annual spend (m.u.)")
    ax.set_title("What each segment buys")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    spend_png = _fig_to_base64(fig)

    # Chart 3: elbow curve with silhouette on a second axis
    fig, ax1 = plt.subplots(figsize=(7, 3.4))
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

    def dot(seg):
        return "#%02x%02x%02x" % tuple(int(255 * c) for c in colors[seg % 10])

    seg_rows = ""
    for seg, r in profile.iterrows():
        seg_rows += (
            f"<tr><td><span class='dot' style='background:{dot(seg)}'></span>"
            f"<b>{r['Segment_Name']}</b></td><td>{int(r['Customers'])} ({r['Share_pct']}%)</td>"
            f"<td>{_money(r['Revenue'])} ({r['Revenue_pct']}%)</td><td>{_money(r['Avg_Spend'])}</td>"
            f"<td>{r['Signature_Category']}</td><td>{int(r['Customers_Below_Median'])}</td>"
            f"<td><b>{_money(r['Opportunity'])}</b></td></tr>")

    target_rows = ""
    for _, t in targets.sort_values("Gap", ascending=False).head(15).iterrows():
        target_rows += (
            f"<tr><td>#{int(t['Customer_ID'])}</td>"
            f"<td><span class='dot' style='background:{dot(int(t['Segment']))}'></span>"
            f"{t['Segment_Name']}</td><td>{t['Region']}</td><td>{t['Category']}</td>"
            f"<td>{_money(t['Current_Spend'])}</td><td>{_money(t['Peer_Median'])}</td>"
            f"<td><b>{_money(t['Gap'])}</b></td></tr>")

    sil = sweep["silhouette"][sweep["k"].index(k)]
    best = profile.sort_values("Opportunity", ascending=False).iloc[0]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Wholesale Customer Segments</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 1000px; margin: 32px auto;
        padding: 0 16px; color: #1f2937; }}
 h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 32px; }} .sub {{ color: #6b7280; margin-top: 0; }}
 .cards {{ display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }}
 .card {{ flex: 1; min-width: 150px; background: #f3f4f6; border-radius: 10px; padding: 14px; }}
 .card b {{ display: block; font-size: 24px; color: #2563eb; }}
 .callout {{ background: #ecfdf5; border-left: 4px solid #16a34a; padding: 12px 16px;
            border-radius: 6px; margin: 16px 0; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
 th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }} th {{ background: #f9fafb; }}
 .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
 img {{ max-width: 100%; margin: 12px 0; }}
</style></head><body>
<h1>Wholesale Customer Segments</h1>
<p class="sub">Generated by the Airflow DAG <code>Wholesale_Customer_Segmentation</code>
 on {datetime.now():%Y-%m-%d %H:%M} · amounts in monetary units (m.u.) per year</p>
<div class="cards">
 <div class="card"><b>{int(profile['Customers'].sum())}</b>customers</div>
 <div class="card"><b>{_money(profile['Revenue'].sum())}</b>annual revenue</div>
 <div class="card"><b>{k}</b>segments (silhouette {sil:.2f})</div>
 <div class="card"><b>{_money(profile['Opportunity'].sum())}</b>growth opportunity</div>
</div>
<div class="callout">Biggest opportunity: <b>{best['Segment_Name']}</b>. {int(best['Customers_Below_Median'])}
 {best['Main_Channel']} customers in this segment spend less than their peers on
 <b>{best['Signature_Category']}</b>, worth about <b>{_money(best['Opportunity'])}</b> m.u. a year
 if they caught up to the peer median.</div>

<h2>Segments</h2>
<table><tr><th>Segment</th><th>Customers</th><th>Revenue</th><th>Avg spend</th>
<th>Signature category</th><th>Below median</th><th>Opportunity</th></tr>{seg_rows}</table>
<img src="data:image/png;base64,{share_png}" alt="Customers vs revenue by segment">
<img src="data:image/png;base64,{spend_png}" alt="Spend by segment">

<h2>Who to call first</h2>
<p class="sub">Customers with the largest gap between their spend and their peers' median in the
 segment's signature category. Peers are customers in the same segment and the same channel
 (restaurants vs. restaurants, shops vs. shops). Full list per segment in
 <code>promotion_targets.csv</code>.</p>
<table><tr><th>Customer</th><th>Segment</th><th>Region</th><th>Category</th>
<th>Current spend</th><th>Peer median</th><th>Gap</th></tr>{target_rows}</table>

<h2>How the number of segments was chosen</h2>
<img src="data:image/png;base64,{elbow_png}" alt="Elbow curve and silhouette scores">
</body></html>"""
    (OUTPUT_DIR / "segment_report.html").write_text(html, encoding="utf-8")