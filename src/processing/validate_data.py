from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class ColumnRule:
    dtype: Optional[str] = None          # expected pandas dtype (e.g. "int64", "float64", "object")
    max_null_frac: Optional[float] = None  # e.g. 0.2 means at most 20% nulls
    non_negative: bool = False           # enforce >= 0 for numeric columns


@dataclass
class DatasetRules:
    name: str
    path: Path
    required_columns: Dict[str, ColumnRule]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _check_columns_exist(df: pd.DataFrame, rules: DatasetRules, errors: List[str]) -> None:
    missing = [col for col in rules.required_columns if col not in df.columns]
    if missing:
        errors.append(f"[{rules.name}] Missing required columns: {missing}")


def _check_column_types(df: pd.DataFrame, rules: DatasetRules, warnings: List[str]) -> None:
    for col, rule in rules.required_columns.items():
        if rule.dtype is None or col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if actual != rule.dtype:
            warnings.append(
                f"[{rules.name}] Column '{col}' dtype is {actual}, expected {rule.dtype}"
            )


def _check_null_rates(df: pd.DataFrame, rules: DatasetRules, warnings: List[str]) -> None:
    n_rows = len(df)
    if n_rows == 0:
        warnings.append(f"[{rules.name}] Dataset has 0 rows.")
        return

    for col, rule in rules.required_columns.items():
        if rule.max_null_frac is None or col not in df.columns:
            continue
        null_frac = df[col].isna().mean()
        if null_frac > rule.max_null_frac:
            warnings.append(
                f"[{rules.name}] Column '{col}' null fraction {null_frac:.2%} "
                f"exceeds threshold {rule.max_null_frac:.2%}"
            )


def _check_value_ranges(df: pd.DataFrame, rules: DatasetRules, errors: List[str]) -> None:
    for col, rule in rules.required_columns.items():
        if not rule.non_negative or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if (df[col] < 0).any():
            errors.append(
                f"[{rules.name}] Column '{col}' contains negative values."
            )


def validate_dataset(rules: DatasetRules) -> bool:
    print("=" * 80)
    print(f"Validating dataset: {rules.name}")
    print("=" * 80)

    df = _load_csv(rules.path)
    print(f"Loaded {rules.name}: shape={df.shape}")

    errors: List[str] = []
    warnings: List[str] = []

    _check_columns_exist(df, rules, errors)
    _check_column_types(df, rules, warnings)
    _check_null_rates(df, rules, warnings)
    _check_value_ranges(df, rules, errors)

    if errors:
        print("\n❌ ERRORS:")
        for msg in errors:
            print(" -", msg)
    else:
        print("\n✅ No blocking errors found.")

    if warnings:
        print("\n⚠️ WARNINGS:")
        for msg in warnings:
            print(" -", msg)
    else:
        print("\n✅ No warnings above configured thresholds.")

    print()
    return not errors


def run_validation() -> None:
    """Run validation for all three raw FEMA datasets."""

    # Rules based on your EDA and project brief
    decl_rules = DatasetRules(
        name="Disaster Declarations Summaries",
        path=RAW_DATA_DIR / "disaster_declarations_summaries.csv",
        required_columns={
            "disasterNumber": ColumnRule(dtype="int64", max_null_frac=0.0),
            "state": ColumnRule(dtype="object", max_null_frac=0.0),
            "incidentType": ColumnRule(dtype="object", max_null_frac=0.0),
            "declarationDate": ColumnRule(dtype="object", max_null_frac=0.0),
            "incidentBeginDate": ColumnRule(dtype="object", max_null_frac=0.0),
            # These are allowed to have nulls, so we omit max_null_frac
            "incidentEndDate": ColumnRule(dtype="object"),
        },
    )

    pa_rules = DatasetRules(
        name="Public Assistance Funded Projects Details",
        path=RAW_DATA_DIR / "public_assistance_funded_projects_details.csv",
        required_columns={
             "disasterNumber": ColumnRule(dtype="int64", max_null_frac=0.0),
             "stateAbbreviation": ColumnRule(dtype="object", max_null_frac=0.0),
             "incidentType": ColumnRule(dtype="object", max_null_frac=0.0),
                # keep type + null checks, but allow negatives (de-obligations)
             "totalObligated": ColumnRule(dtype="float64", max_null_frac=0.0),
             "federalShareObligated": ColumnRule(dtype="float64", max_null_frac=0.0),
        },
    )

    summ_rules = DatasetRules(
        name="FEMA Web Disaster Summaries",
        path=RAW_DATA_DIR / "fema_web_disaster_summaries.csv",
        required_columns={
            "disasterNumber": ColumnRule(dtype="int64", max_null_frac=0.0),
            # allow negatives here too; they reflect adjustments
            "totalObligatedAmountPa": ColumnRule(dtype="float64"),
            "totalObligatedAmountCatAb": ColumnRule(dtype="float64"),
            "totalObligatedAmountCatC2g": ColumnRule(dtype="float64"),
            "totalObligatedAmountHmgp": ColumnRule(dtype="float64"),
        },
    )


    all_ok = True
    for rules in [decl_rules, pa_rules, summ_rules]:
        ok = validate_dataset(rules)
        all_ok = all_ok and ok

    if all_ok:
        print("🎉 DATA VALIDATION PASSED for all datasets.")
    else:
        print("❗ DATA VALIDATION FAILED. See errors above.")
        raise SystemExit(1)


if __name__ == "__main__":
    run_validation()