"""
Unit tests for Phase 12C Component-Based Context Completeness Parser.

Verifies:
1. Complete standalone clinical contexts (including single components, lowercase initial, History of prefixes).
2. Incomplete fragmented contexts (dangling conjunctions, dangling prepositions, trailing punctuation fragments).
3. Empty/None/whitespace inputs.
4. Explicit verification that lowercase initial alone and "History of" alone do NOT imply incomplete.
"""

import pytest
import pandas as pd

from src.evidence_state.context_completeness import (
    assess_context_completeness,
    parse_context_completeness,
    ContextCompletenessResult,
)


@pytest.mark.parametrize(
    "text",
    [
        "Chest pain.",
        "Dyspnea.",
        "History of CHF.",
        "History of asthma.",
        "History of mitral valve prolapse.",
        "History of stent placement 7+ years ago.",
        "Preoperative evaluation.",
        "Nausea and vomiting.",
        "cough and fever.",
    ],
)
def test_complete_contexts(text: str) -> None:
    """Verify that legitimate complete standalone clinical contexts return True."""
    res = parse_context_completeness(text)
    assert res.is_complete is True, f"Expected complete for '{text}', got False ({res.rule_triggered})"
    assert res.rule_triggered == "none"
    assert assess_context_completeness(text) is True


@pytest.mark.parametrize(
    "text,expected_rule",
    [
        ("history of hypertension and", "dangling_conjunction"),
        ("preop for", "dangling_preposition"),
        ("eval for", "dangling_preposition"),
        ("history of", "dangling_preposition"),
        ("hypoxia and", "dangling_conjunction"),
        ("chest pain,", "trailing_punctuation_fragment"),
    ],
)
def test_incomplete_fragmented_contexts(text: str, expected_rule: str) -> None:
    """Verify that structurally fragmented contexts return False with correct rule_triggered."""
    res = parse_context_completeness(text)
    assert res.is_complete is False, f"Expected incomplete for '{text}', got True"
    assert res.rule_triggered == expected_rule
    assert assess_context_completeness(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\t\n",
        None,
    ],
)
def test_empty_or_null_inputs(text: str) -> None:
    """Verify empty/null/whitespace inputs return False with empty_input rule."""
    res = parse_context_completeness(text)
    assert res.is_complete is False
    assert res.rule_triggered == "empty_input"
    assert assess_context_completeness(text) is False


def test_explicit_lowercase_initial_invariance() -> None:
    """Explicit test verifying that lowercase initial character alone does NOT trigger incomplete."""
    text = "cough and fever."
    assert text[0].islower()
    res = parse_context_completeness(text)
    assert res.is_complete is True
    assert res.rule_triggered == "none"


def test_explicit_history_prefix_invariance() -> None:
    """Explicit test verifying that 'History of...' prefix alone does NOT trigger incomplete."""
    text = "History of mitral valve prolapse."
    assert text.startswith("History of")
    res = parse_context_completeness(text)
    assert res.is_complete is True
    assert res.rule_triggered == "none"
