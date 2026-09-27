from src.evaluation.claim_patterns import detect_polarity


def _start(sentence: str, pattern: str) -> int:
    return sentence.lower().find(pattern)


def test_explicit_coordinated_negation():

    sentence = "No pneumothorax or pleural effusion."

    pneumothorax_start = _start(sentence, "pneumothorax")
    effusion_start = _start(sentence, "pleural effusion")

    assert detect_polarity(
        sentence,
        pneumothorax_start,
    ) == "NEGATED"

    assert detect_polarity(
        sentence,
        effusion_start,
    ) == "NEGATED"


def test_typical_findings_negation():

    sentence = "No typical findings of pulmonary edema."
    start = _start(sentence, "pulmonary edema")

    assert detect_polarity(sentence, start) == "NEGATED"


def test_lungs_are_clear_is_not_blanket_negation():

    sentence = "The lungs are clear."

    assert "pneumothorax" not in sentence.lower()
    assert "pleural effusion" not in sentence.lower()
    assert "consolidation" not in sentence.lower()


def test_clear_lungs_do_not_negate_unmentioned_finding():
    """
    A 'clear lungs' sentence does not create a pneumothorax
    mention, so polarity is not inferred for that finding.
    """

    from src.evaluation.claim_patterns import (
        extract_raw_claim_records,
    )

    records = extract_raw_claim_records(
        "The lungs are clear."
    )

    findings = {row["finding"] for row in records}

    assert "pneumothorax" not in findings
    assert "pleural effusion" not in findings
    assert "consolidation" not in findings


def test_expanded_patterns_and_polarity():
    from src.evaluation.claim_patterns import (
        extract_raw_claim_records,
    )

    # 1. "There is a right upper lobe opacity." -> airspace opacity / AFFIRMED
    records = extract_raw_claim_records("There is a right upper lobe opacity.")
    assert len(records) == 1
    assert records[0]["finding"] == "airspace opacity"
    assert records[0]["polarity"] == "AFFIRMED"

    # 2. "No focal airspace opacity is identified." -> airspace opacity / NEGATED
    records = extract_raw_claim_records("No focal airspace opacity is identified.")
    assert len(records) == 1
    assert records[0]["finding"] == "airspace opacity"
    assert records[0]["polarity"] == "NEGATED"

    # 3. "There is right upper lobe pneumonia." -> pneumonia / AFFIRMED
    records = extract_raw_claim_records("There is right upper lobe pneumonia.")
    assert len(records) == 1
    assert records[0]["finding"] == "pneumonia"
    assert records[0]["polarity"] == "AFFIRMED"

    # 4. "No pneumonia." -> pneumonia / NEGATED
    records = extract_raw_claim_records("No pneumonia.")
    assert len(records) == 1
    assert records[0]["finding"] == "pneumonia"
    assert records[0]["polarity"] == "NEGATED"

    # 5. "No infiltrate." -> infiltrate / NEGATED
    records = extract_raw_claim_records("No infiltrate.")
    assert len(records) == 1
    assert records[0]["finding"] == "infiltrate"
    assert records[0]["polarity"] == "NEGATED"

    # 6. "No displaced fracture is seen." -> fracture / NEGATED
    records = extract_raw_claim_records("No displaced fracture is seen.")
    assert len(records) == 1
    assert records[0]["finding"] == "fracture"
    assert records[0]["polarity"] == "NEGATED"

