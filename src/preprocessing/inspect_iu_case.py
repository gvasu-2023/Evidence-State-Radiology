from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PARQUET_PATH = Path(
    "external_data/IU-Xray/data/"
    "train-00000-of-00005-c80629f6027d8eb1.parquet"
)

OUTPUT_DIR = Path("data/samples/iu_test")

IMAGE_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("IU X-RAY CASE INSPECTION")
    print("=" * 70)

    # Create output directories
    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load one Parquet shard
    # --------------------------------------------------

    print("\nLoading dataset shard...")

    df = pd.read_parquet(PARQUET_PATH)

    print(f"Rows available: {len(df)}")

    # --------------------------------------------------
    # Select first case
    # --------------------------------------------------

    row = df.iloc[0]

    uid = row["uid"]

    print(f"\nSelected UID: {uid}")

    # --------------------------------------------------
    # Extract textual information
    # --------------------------------------------------

    indication = row["indication"]
    comparison = row["comparison"]
    findings = row["findings"]
    impression = row["impression"]

    print("\n" + "=" * 70)
    print("CLINICAL CONTEXT")
    print("=" * 70)

    print(indication)

    print("\n" + "=" * 70)
    print("FINDINGS")
    print("=" * 70)

    print(findings)

    print("\n" + "=" * 70)
    print("IMPRESSION")
    print("=" * 70)

    print(impression)

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    metadata_file = METADATA_DIR / f"IU_{uid}.txt"

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(f"UID: {uid}\n\n")

        f.write("INDICATION:\n")
        f.write(str(indication))
        f.write("\n\n")

        f.write("COMPARISON:\n")
        f.write(str(comparison))
        f.write("\n\n")

        f.write("FINDINGS:\n")
        f.write(str(findings))
        f.write("\n\n")

        f.write("IMPRESSION:\n")
        f.write(str(impression))
        f.write("\n")

    print(
        f"\nMetadata saved to: {metadata_file}"
    )

    # --------------------------------------------------
    # Save frontal image
    # --------------------------------------------------

    frontal_image = row["img_frontal"]

    if frontal_image is not None:

        frontal_path = (
            IMAGE_DIR / f"IU_{uid}_frontal.jpg"
        )

        with open(
            frontal_path,
            "wb",
        ) as f:

            f.write(frontal_image)

        print(
            f"Frontal image saved to: {frontal_path}"
        )

    # --------------------------------------------------
    # Save lateral image
    # --------------------------------------------------

    lateral_image = row["img_lateral"]

    if lateral_image is not None:

        lateral_path = (
            IMAGE_DIR / f"IU_{uid}_lateral.jpg"
        )

        with open(
            lateral_path,
            "wb",
        ) as f:

            f.write(lateral_image)

        print(
            f"Lateral image saved to: {lateral_path}"
        )

    print("\n" + "=" * 70)
    print("CASE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()