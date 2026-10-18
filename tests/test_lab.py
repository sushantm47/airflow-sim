"""
Tests for dags/src/lab.py. They run without Docker or Airflow.

They use a small synthetic customer table with the same columns as the UCI file,
so CI does not depend on the downloaded dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src import lab


def make_customers(n=200, seed=0):
    """Two clearly different customer groups: restaurants vs shops."""
    rng = np.random.default_rng(seed)
    channel = rng.choice([1, 2], n, p=[0.6, 0.4])
    centers = {1: [9.5, 7.0, 7.3, 8.0, 5.5, 7.0], 2: [8.0, 9.3, 9.8, 6.5, 9.0, 7.0]}
    spend = np.expm1([rng.normal(centers[c], 0.4) for c in channel]).round().astype(int)
    df = pd.DataFrame(spend, columns=lab.SPENDING_COLUMNS)
    df.insert(0, "Region", rng.choice([1, 2, 3], n))
    df.insert(0, "Channel", channel)
    return df


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    data_file = tmp_path / "customers.csv"
    make_customers().to_csv(data_file, index=False)
    monkeypatch.setattr(lab, "DATA_FILE", data_file)
    monkeypatch.setattr(lab, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(lab, "OUTPUT_DIR", tmp_path / "out")
    return tmp_path


@pytest.fixture
def sweep():
    X = lab.data_preprocessing(lab.load_data())
    return lab.build_save_model(X, "model.sav", max_k="6")  # string, like an Airflow template


# --------------------------------------------------------------- validation

def test_validation_passes_on_good_data():
    assert lab.validate_data(lab.load_data())["rows"] == 200


@pytest.mark.parametrize("break_data, message", [
    (lambda df: df.drop(columns=["Milk"]), "missing columns"),
    (lambda df: df.head(50), "only 50 rows"),
    (lambda df: df.assign(Fresh=df["Fresh"].where(df.index != 3)), "null"),
    (lambda df: df.assign(Grocery=-df["Grocery"]), "negative"),
    (lambda df: df.assign(Channel=7), "Channel"),
])
def test_validation_catches_bad_data(break_data, message):
    bad = lab._encode(break_data(make_customers()))
    with pytest.raises(ValueError, match=message):
        lab.validate_data(bad)


# --------------------------------------------------------------- modelling

def test_preprocessing_is_standardized():
    X = lab._decode(lab.data_preprocessing(lab.load_data()))
    assert X.shape == (200, 6)
    assert np.allclose(X.mean(axis=0), 0) and np.allclose(X.std(axis=0), 1)


def test_sweep_shape_and_saved_models(sweep, sandbox):
    assert sweep["k"] == [1, 2, 3, 4, 5, 6]
    assert sweep["silhouette"][0] is None
    assert sweep["sse"] == sorted(sweep["sse"], reverse=True)
    assert (sandbox / "model" / "model.sav").exists()


def test_elbow_finds_two_groups(sweep):
    assert lab.load_model_elbow("model.sav", sweep) == 2


# --------------------------------------------------------------- quality gate

def test_gate_passes_well_separated_segments(sweep):
    assert lab.quality_gate(sweep, 2, "0.15") == "segment_customers_task"


def test_gate_blocks_when_threshold_too_high(sweep):
    assert lab.quality_gate(sweep, 2, 0.99) == "flag_low_quality_task"


def test_gate_blocks_single_segment(sweep):
    assert lab.quality_gate(sweep, 1, 0.0) == "flag_low_quality_task"


# --------------------------------------------------------------- outputs

def test_segments_and_report(sweep, sandbox):
    result = lab.segment_customers("model.sav", 2, sweep)
    out = sandbox / "out"
    assert result["segments"] == 2 and result["opportunity"] > 0

    customers = pd.read_csv(out / "customer_segments.csv")
    assert len(customers) == 200 and customers["Segment"].nunique() == 2
    assert customers["Customer_ID"].tolist() == list(range(1, 201))

    profiles = pd.read_csv(out / "segment_profiles.csv")
    assert profiles["Customers"].sum() == 200
    assert set(profiles["Main_Channel"]) == {"Horeca", "Retail"}
    assert profiles["Revenue_pct"].sum() == pytest.approx(100, abs=0.2)
    assert profiles["Revenue"].sum() == customers["Annual_Spend"].sum()

    targets = pd.read_csv(out / "promotion_targets.csv")
    assert (targets["Gap"] > 0).all()
    assert targets.groupby("Segment").size().max() <= 10
    assert set(targets["Region"]) <= {"Lisbon", "Oporto", "Other"}

    html = (out / "segment_report.html").read_text(encoding="utf-8")
    assert "Who to call first" in html and html.count("data:image/png") == 3


def test_promotion_targets_are_share_of_wallet_gaps():
    df = pd.DataFrame({
        "Customer_ID": [1, 2, 3, 4, 5, 6],
        "Segment": [0, 0, 0, 0, 1, 1],
        "Segment_Name": ["A"] * 4 + ["B"] * 2,
        "Channel": [1, 1, 1, 2, 2, 2],   # customer 4 is a shop inside a restaurant segment
        "Region": [1, 2, 3, 1, 1, 1],
        "Fresh": [100, 50, 10, 0, 0, 0],
        "Grocery": [0, 0, 0, 0, 80, 20],
    })
    targets, opportunity = lab.find_promotion_targets(
        df, {0: "Fresh", 1: "Grocery"}, {0: 1, 1: 2}, top_n=1)
    # segment 0 peers = restaurants 1-3, median 50 -> customer 3 is 40 below
    # segment 1 median Grocery = 50 -> customer 6 is 30 below
    assert targets["Customer_ID"].tolist() == [3, 6]
    assert targets["Gap"].tolist() == [40, 30]
    assert 4 not in targets["Customer_ID"].tolist()   # the shop is not compared with restaurants
    assert opportunity.loc[0, "Opportunity"] == 40 and opportunity.loc[1, "Opportunity"] == 30
    assert opportunity.loc[0, "Customers_Below_Median"] == 1


def test_targets_match_segment_main_channel(sweep, sandbox):
    lab.segment_customers("model.sav", 2, sweep)
    out = sandbox / "out"
    targets = pd.read_csv(out / "promotion_targets.csv")
    profiles = pd.read_csv(out / "segment_profiles.csv").set_index("Segment")
    for _, t in targets.iterrows():
        assert t["Channel"] == profiles.loc[t["Segment"], "Main_Channel"]


def test_segment_names_reflect_spending():
    profile = pd.DataFrame(
        [[100, 10, 10, 50, 5, 10, "Horeca"], [10, 80, 90, 5, 60, 10, "Retail"]],
        columns=lab.SPENDING_COLUMNS + ["Main_Channel"],
    )
    names = lab.name_segments(profile, profile[lab.SPENDING_COLUMNS].mean())
    assert names[0].startswith("Horeca: Fresh produce +")
    assert names[1].startswith("Retail: Cleaning & paper +")