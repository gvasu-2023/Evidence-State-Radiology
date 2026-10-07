"""Strict, inference-free access to the frozen Phase 26D demo data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENERATION_PATH = ROOT / "results/maira2_contrastive/phase26d_evaluation/phase26d_evaluation_generation.csv"
MASTER_PATH = ROOT / "results/tables/phase26d_master/phase26d_master_evaluation.csv"
CONDITIONS = (
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
)
LAMBDAS = (0.0, 0.25)
REQUIRED_GENERATION_COLUMNS = {
    "sample_id", "uid", "condition", "evidence_state", "perturbed_context",
    "frontal_image", "lambda", "report", "num_tokens", "runtime_seconds", "status",
}


class DemoDataError(ValueError):
    """Raised when frozen demo data cannot be read safely."""


def safe_text(value: Any) -> str:
    """Return a display-safe string, treating CSV nulls as empty text."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_generation_records(path: Path = GENERATION_PATH) -> pd.DataFrame:
    """Load the frozen evaluation CSV and reject malformed rows explicitly."""
    try:
        records = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DemoDataError(f"Could not read frozen Phase 26D generation data: {exc}") from exc

    missing_columns = sorted(REQUIRED_GENERATION_COLUMNS - set(records.columns))
    if missing_columns:
        raise DemoDataError(f"Generation CSV is missing required columns: {', '.join(missing_columns)}")
    if records.empty:
        raise DemoDataError("Generation CSV contains no evaluation records.")

    try:
        records["uid"] = pd.to_numeric(records["uid"], errors="raise").astype(int)
        records["lambda"] = pd.to_numeric(records["lambda"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise DemoDataError(f"Generation CSV has a malformed UID or lambda value: {exc}") from exc

    if records["sample_id"].map(safe_text).eq("").any():
        raise DemoDataError("Generation CSV contains a row with an empty sample_id.")
    if records["condition"].map(safe_text).eq("").any():
        raise DemoDataError("Generation CSV contains a row with an empty condition.")
    unknown_conditions = sorted(set(records["condition"].map(safe_text)) - set(CONDITIONS))
    if unknown_conditions:
        raise DemoDataError(f"Generation CSV contains unknown conditions: {unknown_conditions}")
    unknown_lambdas = sorted(set(records["lambda"]) - set(LAMBDAS))
    if unknown_lambdas:
        raise DemoDataError(f"Generation CSV contains unexpected lambda values: {unknown_lambdas}")

    duplicate_mask = records.duplicated(["sample_id", "condition", "lambda"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = records.loc[duplicate_mask, ["sample_id", "condition", "lambda"]]
        examples = duplicate_keys.head(3).to_dict(orient="records")
        raise DemoDataError(f"Generation CSV has duplicate study-condition-lambda rows: {examples}")

    return records


def get_frozen_pair(records: pd.DataFrame, sample_id: str, condition: str) -> dict[float, pd.Series]:
    """Return only the available frozen lambda records; never substitute a report."""
    if sample_id not in set(records["sample_id"].astype(str)):
        return {}
    if condition not in CONDITIONS:
        return {}
    selected = records.loc[
        records["sample_id"].astype(str).eq(str(sample_id))
        & records["condition"].astype(str).eq(condition)
    ]
    return {float(row["lambda"]): row for _, row in selected.iterrows()}


def get_study_context(records: pd.DataFrame, sample_id: str, condition: str) -> str:
    """Return the frozen context for a selected key, or empty text if absent."""
    pair = get_frozen_pair(records, sample_id, condition)
    if not pair:
        return ""
    row = pair.get(0.0)
    if row is None:
        row = pair.get(0.25)
    return safe_text(row.get("perturbed_context"))


def find_quick_demo_sample(records: pd.DataFrame, condition: str) -> str:
    """Choose the lowest-UID available example for a quick demo condition."""
    examples = records.loc[records["condition"].eq(condition), ["sample_id", "uid"]]
    if examples.empty:
        raise DemoDataError(f"No frozen evaluation example exists for condition '{condition}'.")
    examples = examples.sort_values(["uid", "sample_id"])
    return str(examples.iloc[0]["sample_id"])


def resolve_image_path(image_path: Any, root: Path = ROOT) -> Path | None:
    """Resolve the exact CSV image path if it exists; never substitute an image."""
    raw_path = safe_text(image_path)
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    return candidate if candidate.is_file() else None


def load_master_results(path: Path = MASTER_PATH) -> pd.DataFrame:
    """Load frozen Phase 26D master metrics for the results snapshot."""
    try:
        results = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DemoDataError(f"Could not read frozen Phase 26D master results: {exc}") from exc
    required = {"metric", "lambda_0", "lambda_025", "interpretation"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise DemoDataError(f"Phase 26D master table is missing required columns: {', '.join(missing)}")
    return results
