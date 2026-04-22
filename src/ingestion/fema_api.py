from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "disaster_declarations_summaries": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance_funded_projects_details": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "fema_web_disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

PAGE_SIZE = 1000
RATE_LIMIT_DELAY = 0.3
MAX_RETRIES = 5
FRESHNESS_DAYS = 7
REQUEST_TIMEOUT = 60


def _get_response_records(payload: Dict) -> List[Dict]:
    """
    FEMA API responses usually contain one top-level list field
    (e.g. DisasterDeclarationsSummaries, PublicAssistanceFundedProjectsDetails).
    This function finds and returns that list.
    """
    for key, value in payload.items():
        if isinstance(value, list):
            return value
    raise ValueError("No list of records found in FEMA API response.")


def _is_fresh(file_path: Path, max_age_days: int = FRESHNESS_DAYS) -> bool:
    """
    Check whether a local CSV file is newer than the freshness threshold.
    """
    if not file_path.exists():
        return False

    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - modified_time
    return age < timedelta(days=max_age_days)


def _request_with_retry(url: str, params: Dict) -> Dict:
    """
    Make a GET request with retry logic and exponential backoff.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 400:
                print("400 Bad Request details:")
                print("URL:", response.url)
                print("Response text:", response.text[:1000])

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as exc:
            wait_time = 2 ** attempt
            print(
                f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}) for {url}. "
                f"Error: {exc}. Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to fetch data from {url} after {MAX_RETRIES} attempts.")


def _fetch_paginated(url: str, page_size: int = PAGE_SIZE) -> pd.DataFrame:
    """
    Fetch all pages from a FEMA endpoint using $top and $skip pagination.
    Saves partial progress in memory and returns all successful pages collected.
    """
    all_records: List[Dict] = []
    skip = 0

    while True:
        params = {
            "$top": page_size,
            "$skip": skip,
            "$format": "json",
        }

        try:
            payload = _request_with_retry(url, params)
            records = _get_response_records(payload)

            if not records:
                print(f"No more records returned from {url} at skip={skip}.")
                break

            all_records.extend(records)
            print(f"Fetched {len(records)} records from {url} (skip={skip}).")

            if len(records) < page_size:
                print(f"Final page reached for {url}.")
                break

            skip += page_size
            time.sleep(RATE_LIMIT_DELAY)

        except RuntimeError as exc:
            print(f"Stopping pagination early for {url} due to repeated failures.")
            print(f"Last successful skip: {skip}")
            print(f"Records collected so far: {len(all_records)}")
            break

    return pd.DataFrame(all_records)


def fetch_disaster_declarations() -> pd.DataFrame:
    """
    Fetch FEMA Disaster Declarations Summaries.
    """
    return _fetch_paginated(ENDPOINTS["disaster_declarations_summaries"])


def fetch_public_assistance_projects() -> pd.DataFrame:
    """
    Fetch FEMA Public Assistance Funded Projects Details.
    """
    return _fetch_paginated(ENDPOINTS["public_assistance_funded_projects_details"])


def fetch_fema_web_disaster_summaries() -> pd.DataFrame:
    """
    Fetch FEMA Web Disaster Summaries.
    """
    return _fetch_paginated(ENDPOINTS["fema_web_disaster_summaries"])


def save_dataset(df: pd.DataFrame, filename: str) -> Path:
    """
    Save a dataframe to data/raw as CSV.
    """
    file_path = RAW_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df):,} records to {file_path}")
    return file_path


def fetch_and_save_dataset(
    dataset_name: str,
    fetch_function,
    filename: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch a dataset unless a fresh local copy already exists.
    """
    file_path = RAW_DATA_DIR / filename

    if not force_refresh and _is_fresh(file_path):
        print(f"Skipping fetch for {dataset_name}: fresh file already exists at {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} records from existing file.")
        return df

    print(f"Fetching dataset: {dataset_name}")
    df = fetch_function()
    save_dataset(df, filename)
    return df


def run_full_ingestion(force_refresh: bool = False) -> None:
    """
    Run the full ingestion process for all 3 FEMA datasets.
    """
    declarations_df = fetch_and_save_dataset(
        dataset_name="Disaster Declarations Summaries",
        fetch_function=fetch_disaster_declarations,
        filename="disaster_declarations_summaries.csv",
        force_refresh=force_refresh,
    )

    pa_df = fetch_and_save_dataset(
        dataset_name="Public Assistance Funded Projects Details",
        fetch_function=fetch_public_assistance_projects,
        filename="public_assistance_funded_projects_details.csv",
        force_refresh=force_refresh,
    )

    summary_df = fetch_and_save_dataset(
        dataset_name="FEMA Web Disaster Summaries",
        fetch_function=fetch_fema_web_disaster_summaries,
        filename="fema_web_disaster_summaries.csv",
        force_refresh=force_refresh,
    )

    print("\nIngestion complete.")
    print(f"Declarations rows: {len(declarations_df):,}")
    print(f"PA project rows: {len(pa_df):,}")
    print(f"Disaster summary rows: {len(summary_df):,}")


if __name__ == "__main__":
    run_full_ingestion(force_refresh=False)