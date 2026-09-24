from pathlib import Path

import pandas as pd


DATASET_CSV = Path(
    "data/processed/iu_xray/metadata/dataset.csv"
)


def is_missing(value):
    if pd.isna(value):
        return True

    value = str(value).strip()

    return (
        value == ""
        or value.lower() in {
            "none",
            "nan",
            "null",
        }
    )


def main():

    print("=" * 70)
    print("IU X-RAY DATA QUALITY AUDIT")
    print("=" * 70)

    df = pd.read_csv(DATASET_CSV)

    print(f"\nTotal cases: {len(df)}")

    # --------------------------------------------------
    # Check required fields
    # --------------------------------------------------

    required_fields = [
        "sample_id",
        "frontal_image",
        "clinical_context",
        "findings",
        "impression",
    ]

    print("\nRequired-field completeness:")
    print("-" * 70)

    for field in required_fields:

        missing = df[field].apply(
            is_missing
        ).sum()

        available = len(df) - missing

        print(
            f"{field:20s} "
            f"available={available:2d} "
            f"missing={missing:2d}"
        )

    # --------------------------------------------------
    # Check image existence
    # --------------------------------------------------

    print("\nImage existence:")
    print("-" * 70)

    image_exists = []

    for path in df["frontal_image"]:

        exists = Path(path).exists()

        image_exists.append(exists)

    print(
        f"Frontal images found: "
        f"{sum(image_exists)}/{len(image_exists)}"
    )

    # --------------------------------------------------
    # Context lengths
    # --------------------------------------------------

    df["context_length"] = (
        df["clinical_context"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print("\nClinical-context statistics:")
    print("-" * 70)

    print(
        df["context_length"].describe()
    )

    # --------------------------------------------------
    # Report lengths
    # --------------------------------------------------

    df["findings_length"] = (
        df["findings"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["impression_length"] = (
        df["impression"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print("\nReference-text statistics:")
    print("-" * 70)

    print(
        "Findings character count:"
    )

    print(
        df["findings_length"].describe()
    )

    print(
        "\nImpression character count:"
    )

    print(
        df["impression_length"].describe()
    )

    # --------------------------------------------------
    # De-identification markers
    # --------------------------------------------------

    deid_pattern = "XXXX"

    context_deid = (
        df["clinical_context"]
        .fillna("")
        .astype(str)
        .str.contains(
            deid_pattern,
            case=False,
            regex=False,
        )
    )

    findings_deid = (
        df["findings"]
        .fillna("")
        .astype(str)
        .str.contains(
            deid_pattern,
            case=False,
            regex=False,
        )
    )

    impression_deid = (
        df["impression"]
        .fillna("")
        .astype(str)
        .str.contains(
            deid_pattern,
            case=False,
            regex=False,
        )
    )

    print("\nDe-identification markers:")
    print("-" * 70)

    print(
        f"Context containing XXXX: "
        f"{context_deid.sum()}"
    )

    print(
        f"Findings containing XXXX: "
        f"{findings_deid.sum()}"
    )

    print(
        f"Impression containing XXXX: "
        f"{impression_deid.sum()}"
    )

    # --------------------------------------------------
    # Case summary
    # --------------------------------------------------

    print("\nCase summary:")
    print("-" * 70)

    print(
        df[
            [
                "sample_id",
                "clinical_context",
                "impression",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()