from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_data() -> Dict[str, pd.DataFrame]:
    decl_path = RAW_DATA_DIR / "disaster_declarations_summaries.csv"
    pa_path = RAW_DATA_DIR / "public_assistance_funded_projects_details.csv"
    summ_path = RAW_DATA_DIR / "fema_web_disaster_summaries.csv"

    decl = pd.read_csv(decl_path)
    pa = pd.read_csv(pa_path)
    summ = pd.read_csv(summ_path)

    return {"decl": decl, "pa": pa, "summ": summ}


def _prepare_keys(decl: pd.DataFrame, pa: pd.DataFrame, summ: pd.DataFrame):
    # Standardise disasterNumber to string for safe joins
    decl = decl.copy()
    pa = pa.copy()
    summ = summ.copy()

    decl["disasterNumber"] = decl["disasterNumber"].astype(str)
    pa["disasterNumber"] = pa["disasterNumber"].astype(str)
    summ["disasterNumber"] = summ["disasterNumber"].astype(str)

    return decl, pa, summ


def _aggregate_pa_to_disaster(pa: pd.DataFrame) -> pd.DataFrame:
    """Aggregate project-level PA data to disaster level.

    For now we:
      - sum totalObligated and federalShareObligated
      - count projects
      - compute average projectAmount
    """

    # Ensure numeric
    for col in ["totalObligated", "federalShareObligated", "projectAmount"]:
        if col in pa.columns:
            pa[col] = pd.to_numeric(pa[col], errors="coerce")

    grouped = (
        pa.groupby("disasterNumber", as_index=False)
        .agg(
            total_obligated_pa=("totalObligated", "sum"),
            federal_share_pa=("federalShareObligated", "sum"),
            project_count=("projectAmount", "count"),
            avg_project_amount=("projectAmount", "mean"),
        )
    )

    return grouped


def _build_base_disaster_table(decl: pd.DataFrame) -> pd.DataFrame:
    """Create base disaster-level features from declarations."""

    decl = decl.copy()

    # Parse dates
    decl["declarationDate"] = pd.to_datetime(decl["declarationDate"], errors="coerce")
    decl["incidentBeginDate"] = pd.to_datetime(decl["incidentBeginDate"], errors="coerce")
    decl["incidentEndDate"] = pd.to_datetime(decl["incidentEndDate"], errors="coerce")

    # Core identifiers and categorical features
    base = decl[[
        "disasterNumber",
        "state",
        "incidentType",
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "declarationType",
        "region",
    ]].drop_duplicates(subset=["disasterNumber"])

    # Temporal features
    base["declaration_year"] = base["declarationDate"].dt.year
    base["declaration_month"] = base["declarationDate"].dt.month

    # Incident duration in days (fallback: 0 if end date missing)
    base["incident_duration_days"] = (
        (base["incidentEndDate"] - base["incidentBeginDate"])
        .dt.days
        .fillna(0)
        .clip(lower=0)
    )

    # Simple 4-season encoding from month
    def month_to_season(m: float) -> str:
        if np.isnan(m):
            return "Unknown"
        m = int(m)
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Autumn"

    base["season"] = base["declaration_month"].apply(month_to_season)

    return base


def _add_historical_frequency(base: pd.DataFrame) -> pd.DataFrame:
    """Add 5-year rolling disaster count per state based on declaration year.

    This uses only declaration history; for more precise windows you can later refine.
    """
    df = base.copy()

    df = df.sort_values(["state", "declaration_year"])
    counts = []

    for state, group in df.groupby("state"):
        years = group["declaration_year"].values
        hist_counts = []
        for i, year in enumerate(years):
            window_start = year - 5
            mask = (group["declaration_year"] < year) & (group["declaration_year"] >= window_start)
            hist_counts.append(mask.sum())
        counts.extend(hist_counts)

    df["state_5yr_disaster_count"] = counts
    return df


def _add_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add high_cost_incident boolean for selected incident types."""
    high_cost_types = {"Hurricane", "Flood", "Tornado", "Severe Storm", "Severe Storm(s)"}
    df["high_cost_incident"] = df["incidentType"].isin(high_cost_types)
    return df


def run_feature_engineering() -> Path:
    raw = _load_raw_data()
    decl, pa, summ = _prepare_keys(raw["decl"], raw["pa"], raw["summ"])

    # Base disaster features from declarations
    base = _build_base_disaster_table(decl)

    # Aggregate PA project data to disaster level
    pa_agg = _aggregate_pa_to_disaster(pa)

    # Merge base + PA aggregates
    df = base.merge(pa_agg, on="disasterNumber", how="left")

    # Optional: cross-check with FEMA summaries (e.g., keep for future analysis)
    # Bring in totalObligatedAmountPa for now as a reference
    summ_reduced = summ[["disasterNumber", "totalObligatedAmountPa"]].copy()
    df = df.merge(summ_reduced, on="disasterNumber", how="left")

    # Add historical frequency and risk flags
    df = _add_historical_frequency(df)
    df = _add_risk_flag(df)

    # Define modelling target (log-transformed)
    # Use the PA aggregation as the primary target
    df["total_obligated_pa"] = df["total_obligated_pa"].fillna(0)
    df["target_total_obligated"] = df["total_obligated_pa"]
    df["target_log_total_obligated"] = np.log1p(df["target_total_obligated"])

    # Save
    out_path = PROCESSED_DATA_DIR / "processed_disasters.csv"
    df.to_csv(out_path, index=False)

    print(f"Processed dataset saved to: {out_path}")
    print("Final shape:", df.shape)
    return out_path


if __name__ == "__main__":
    run_feature_engineering()