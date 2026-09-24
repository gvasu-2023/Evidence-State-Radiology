from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    "external_data/IU-Xray/data"
)

OUTPUT_DIR = Path(
    "data/processed/iu_xray"
)

IMAGE_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"

NUM_CASES = 20


# ============================================================
# Helper
# ============================================================

def save_bytes(data, path):

    if data is None:
        return False

    with open(path, "wb") as f:
        f.write(data)

    return True


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING IU X-RAY RESEARCH SUBSET")
    print("=" * 70)

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find Parquet shards
    # --------------------------------------------------------

    parquet_files = sorted(
        DATASET_DIR.glob("*.parquet")
    )

    print(
        f"\nFound {len(parquet_files)} Parquet shards."
    )

    if not parquet_files:
        raise FileNotFoundError(
            "No Parquet files found."
        )

    # --------------------------------------------------------
    # Collect cases
    # --------------------------------------------------------

    selected_rows = []

    for parquet_file in parquet_files:

        print(
            f"\nReading: {parquet_file.name}"
        )

        df = pd.read_parquet(
            parquet_file
        )

        print(
            f"Rows: {len(df)}"
        )

        for _, row in df.iterrows():

            # Require the fields needed for our experiment.
            if row["img_frontal"] is None:
                continue

            if row["indication"] is None:
                continue

            if row["findings"] is None:
                continue

            if row["impression"] is None:
                continue

            selected_rows.append(row)

            if len(selected_rows) >= NUM_CASES:
                break

        if len(selected_rows) >= NUM_CASES:
            break

    print(
        f"\nSelected cases: {len(selected_rows)}"
    )

    # --------------------------------------------------------
    # Save cases
    # --------------------------------------------------------

    metadata_rows = []

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):

        uid = str(row["uid"])

        sample_id = f"IU_{uid}"

        print(
            f"\n[{index}/{len(selected_rows)}] "
            f"Processing {sample_id}"
        )

        # ----------------------------------------------------
        # Save frontal image
        # ----------------------------------------------------

        frontal_path = (
            IMAGE_DIR
            / f"{sample_id}_frontal.jpg"
        )

        save_bytes(
            row["img_frontal"],
            frontal_path,
        )

        # ----------------------------------------------------
        # Save lateral image if available
        # ----------------------------------------------------

        lateral_path = None

        if row["img_lateral"] is not None:

            lateral_path = (
                IMAGE_DIR
                / f"{sample_id}_lateral.jpg"
            )

            save_bytes(
                row["img_lateral"],
                lateral_path,
            )

        # ----------------------------------------------------
        # Create metadata record
        # ----------------------------------------------------

        metadata_rows.append(
            {
                "sample_id": sample_id,
                "uid": uid,
                "frontal_image":
                    str(frontal_path),
                "lateral_image":
                    str(lateral_path)
                    if lateral_path
                    else "",
                "clinical_context":
                    str(row["indication"]),
                "comparison":
                    str(row["comparison"]),
                "findings":
                    str(row["findings"]),
                "impression":
                    str(row["impression"]),
                "mesh":
                    str(row["MeSH"]),
                "problems":
                    str(row["Problems"]),
            }
        )

    # --------------------------------------------------------
    # Save metadata CSV
    # --------------------------------------------------------

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_file = (
        METADATA_DIR / "dataset.csv"
    )

    metadata_df.to_csv(
        metadata_file,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SUBSET COMPLETE")
    print("=" * 70)

    print(
        f"Cases saved: {len(metadata_df)}"
    )

    print(
        f"Metadata: {metadata_file}"
    )

    print(
        f"Images: {IMAGE_DIR}"
    )


if __name__ == "__main__":
    main()