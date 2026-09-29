"""Build the image-backed Phase 17 VLM inference eligibility cohort."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "results/tables/phase16_perturbations.csv"
SELECTION_PATH = ROOT / "results/tables/phase16e_selected_studies.csv"
OUTPUT_PATH = ROOT / "results/tables/phase17_vlm_inference_eligibility.csv"

EXPECTED_BENCHMARK_SHA256 = "3d003349baa86c619af16ad4fbebba7b3bd174266e579a2260f7a9fcf4afce53"
EXPECTED_SELECTION_SHA256 = "21144953c8d796a31c937405386d03d52cd933008f765befbc255b2c12d8cd0a"
EXCLUDED_UIDS = {74, 597, 803, 885}
CONDITIONS = {
    "sufficient",
    "syntactic_incomplete",
    "evidentiary_incomplete",
    "irrelevant",
    "conflicting",
    "insufficient",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _image_path_for_row(row: pd.Series, root: Path) -> tuple[str, Path]:
    """Use the benchmark frontal reference, or its canonical expected path if blank."""
    benchmark_ref = _text(row["frontal_image"])
    if benchmark_ref:
        relative_path = benchmark_ref.replace("\\", "/")
    else:
        relative_path = f"external_data/IU-Xray/images/IU_{int(row['uid'])}_frontal.jpg"
    return relative_path, root / Path(relative_path)


def inspect_image(path: Path) -> dict[str, Any]:
    """Apply the Phase 17 image integrity checks to one frontal image."""
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    readable = False
    verified = False
    width = 0
    height = 0

    if exists and size > 0:
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.load()
            readable = width > 0 and height > 0
        except (OSError, ValueError, Image.DecompressionBombError):
            readable = False

        try:
            with Image.open(path) as image:
                image.verify()
            verified = True
        except (OSError, ValueError, Image.DecompressionBombError):
            verified = False

    return {
        "image_available": exists,
        "file_size_bytes": size,
        "image_readable": readable,
        "image_verify_ok": verified,
        "width": width,
        "height": height,
        "integrity_ok": bool(exists and size > 0 and readable and verified and width > 0 and height > 0),
    }


def _validate_benchmark(benchmark: pd.DataFrame) -> None:
    required = {"sample_id", "uid", "condition", "frontal_image", "lateral_image"}
    missing = required - set(benchmark.columns)
    if missing:
        raise ValueError(f"Benchmark is missing required columns: {sorted(missing)}")
    if len(benchmark) != 600:
        raise ValueError(f"Expected 600 benchmark rows, found {len(benchmark)}")
    if benchmark["uid"].nunique() != 100:
        raise ValueError(f"Expected 100 unique benchmark UIDs, found {benchmark['uid'].nunique()}")
    if set(benchmark["condition"].dropna().astype(str)) != CONDITIONS:
        raise ValueError("Benchmark condition labels do not match the six expected states")
    if benchmark["condition"].value_counts().to_dict() != {condition: 100 for condition in CONDITIONS}:
        raise ValueError("Expected exactly 100 benchmark rows per condition")
    if benchmark.duplicated(["uid", "condition"]).any():
        raise ValueError("Benchmark contains duplicate (uid, condition) rows")
    if not benchmark.groupby("uid")["condition"].nunique().eq(6).all():
        raise ValueError("Every UID must have exactly six conditions")


def build_eligibility_records(
    benchmark_path: Path = BENCHMARK_PATH,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Create one integrity-backed inference eligibility row per benchmark row."""
    benchmark = pd.read_csv(benchmark_path)
    _validate_benchmark(benchmark)

    image_checks: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for _, row in benchmark.iterrows():
        uid = int(row["uid"])
        image_path, absolute_path = _image_path_for_row(row, root)
        if image_path not in image_checks:
            image_checks[image_path] = inspect_image(absolute_path)
        checks = image_checks[image_path]

        eligible = bool(checks["integrity_ok"])
        if not eligible and uid not in EXCLUDED_UIDS:
            raise ValueError(
                f"UID {uid} is not an approved exclusion but its frontal image failed integrity checks: {image_path}"
            )
        if uid in EXCLUDED_UIDS and eligible:
            raise ValueError(f"UID {uid} is expected to be image-unavailable but passed image integrity checks")

        records.append(
            {
                "sample_id": _text(row["sample_id"]),
                "uid": uid,
                "condition": _text(row["condition"]),
                "image_path": image_path,
                "benchmark_frontal_image": _text(row["frontal_image"]),
                "lateral_image_path": _text(row["lateral_image"]),
                **{key: checks[key] for key in (
                    "image_available", "file_size_bytes", "image_readable", "image_verify_ok", "width", "height"
                )},
                "eligible_for_vlm": eligible,
                "exclusion_reason": "" if eligible else "frontal_image_unavailable",
            }
        )

    eligibility = pd.DataFrame(records)
    _validate_eligibility(eligibility)
    return eligibility


def _validate_eligibility(eligibility: pd.DataFrame) -> None:
    if len(eligibility) != 600:
        raise ValueError(f"Expected 600 eligibility rows, found {len(eligibility)}")
    excluded = eligibility.loc[~eligibility["eligible_for_vlm"]]
    if len(excluded) != 24:
        raise ValueError(f"Expected 24 excluded rows, found {len(excluded)}")
    if set(excluded["uid"].astype(int)) != EXCLUDED_UIDS:
        raise ValueError(f"Unexpected excluded UID set: {sorted(excluded['uid'].astype(int).unique())}")
    if not excluded["exclusion_reason"].eq("frontal_image_unavailable").all():
        raise ValueError("Excluded records must use exclusion_reason=frontal_image_unavailable")
    if excluded.groupby("uid").size().to_dict() != {uid: 6 for uid in EXCLUDED_UIDS}:
        raise ValueError("Each unavailable UID must contribute exactly six excluded records")
    eligible = eligibility.loc[eligibility["eligible_for_vlm"]]
    if len(eligible) != 576:
        raise ValueError(f"Expected 576 eligible rows, found {len(eligible)}")
    if eligible["condition"].value_counts().to_dict() != {condition: 96 for condition in CONDITIONS}:
        raise ValueError("Eligible cohort must contain exactly 96 records per condition")
    if not eligible["image_available"].all() or not eligible["image_readable"].all():
        raise ValueError("Every eligible record must have an available, readable image")
    if not eligible["image_verify_ok"].all():
        raise ValueError("Every eligible image must pass PIL.verify()")
    if not eligible["file_size_bytes"].gt(0).all() or not eligible[["width", "height"]].gt(0).all().all():
        raise ValueError("Every eligible image must be non-empty and have positive dimensions")
    paths = eligible["image_path"].str.lower()
    lateral_paths = eligible["lateral_image_path"].fillna("").str.lower()
    if paths.str.contains("_lateral.").any() or paths.str.contains("_frontal.").eq(False).any():
        raise ValueError("An eligible record does not reference a frontal image")
    if (paths == lateral_paths).any():
        raise ValueError("An eligible record points to its lateral image")


def prepare_eligibility_artifact(
    benchmark_path: Path = BENCHMARK_PATH,
    selection_path: Path = SELECTION_PATH,
    output_path: Path = OUTPUT_PATH,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Build and write the artifact after verifying both frozen input hashes."""
    benchmark_hash = sha256_file(benchmark_path)
    selection_hash = sha256_file(selection_path)
    if benchmark_hash != EXPECTED_BENCHMARK_SHA256:
        raise ValueError(f"Frozen benchmark hash mismatch: {benchmark_hash}")
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise ValueError(f"Frozen selection hash mismatch: {selection_hash}")

    eligibility = build_eligibility_records(benchmark_path=benchmark_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eligibility.to_csv(output_path, index=False)

    if sha256_file(benchmark_path) != benchmark_hash or sha256_file(selection_path) != selection_hash:
        raise RuntimeError("A frozen input file changed while building the eligibility artifact")
    return eligibility


def main() -> None:
    eligibility = prepare_eligibility_artifact()
    print(f"benchmark_records: {len(eligibility)}")
    print(f"eligible_records: {int(eligibility['eligible_for_vlm'].sum())}")
    print(f"excluded_records: {int((~eligibility['eligible_for_vlm']).sum())}")
    print(f"eligible_by_condition: {eligibility.loc[eligibility.eligible_for_vlm, 'condition'].value_counts().sort_index().to_dict()}")
    print(f"excluded_by_uid: {eligibility.loc[~eligibility.eligible_for_vlm].groupby('uid').size().to_dict()}")
    print(f"output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
