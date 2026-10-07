"""Local, inference-free interactive demonstrator for the Phase 28A workflow."""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Streamlit can execute this file with ``demo/`` as the script directory.
# Anchor imports and data access to the repository instead of the process CWD.
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import streamlit as st

from demo.data import (
    CONDITIONS,
    DemoDataError,
    find_quick_demo_sample,
    get_frozen_pair,
    get_study_context,
    load_generation_records,
    load_master_results,
    resolve_image_path,
    safe_text,
)
from src.evidence_state.analyzer import EvidenceAssessment, EvidenceStateAnalyzer
from src.evidence_state.context_completeness import parse_context_completeness
from src.gating.reliability_gate import ReliabilityGate


st.set_page_config(
    page_title="Evidence-State Radiology | Phase 28A",
    page_icon="XR",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --ink: #14283d;
        --muted: #42576b;
        --navy: #123b5d;
        --blue: #176b87;
        --line: #d8e1e9;
        --paper: #ffffff;
        --canvas: #f2f6fa;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--canvas) !important;
        color: var(--ink) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stMain"] { background: var(--canvas) !important; }
    .block-container { max-width: 1660px; padding: .8rem 1.5rem 1.1rem; }
    .stMarkdown, .stText, .stCaption, p, label, li,
    [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] p {
        color: var(--ink) !important;
    }
    h1, h2, h3, h4, [data-testid="stSubheader"] {
        color: #102f4a !important;
        letter-spacing: -.015em;
    }
    h1 { margin: .1rem 0 .15rem !important; font-size: 2rem !important; }
    h2, h3 { margin: .3rem 0 .4rem !important; }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
    [data-testid="stWidgetLabel"] p, label { color: #203b53 !important; font-weight: 600 !important; }
    [data-baseweb="select"] > div, [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input {
        background: #ffffff !important;
        color: #14283d !important;
        border-color: #9aaebe !important;
    }
    [data-baseweb="textarea"] textarea { line-height: 1.45; }
    [data-testid="stCheckbox"] label p { color: #203b53 !important; }
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p { color: #203b53 !important; }
    div[data-testid="stButton"] > button {
        min-height: 2.8rem;
        border: 1px solid #0e3655;
        border-radius: 9px;
        background: #123b5d;
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: .01em;
        box-shadow: 0 2px 5px rgba(17, 46, 69, .13);
    }
    div[data-testid="stButton"] > button p { color: #ffffff !important; }
    div[data-testid="stButton"] > button:hover {
        border-color: #176b87;
        background: #176b87;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] { background: #ffffff; border: 1px solid var(--line); border-radius: 9px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
    .title-row { display: flex; align-items: baseline; gap: .8rem; margin-bottom: .1rem; }
    .title-tag { color: #176b87; font-size: .8rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
    .how-title { color: #203b53; font-size: .76rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; margin: .3rem 0 .35rem; }
    .flow { display: grid; grid-template-columns: 1.15fr 1.15fr .9fr .9fr .7fr 1.15fr; gap: .45rem; margin: 0 0 .4rem; }
    .flow-step { position: relative; min-height: 3.35rem; display: flex; align-items: center; justify-content: center;
        padding: .45rem .55rem; border: 1px solid #c7d6e2; border-radius: 9px; background: #ffffff;
        color: #173b57; font-size: .77rem; line-height: 1.2; font-weight: 800; text-align: center; }
    .flow-step:not(:last-child)::after { content: "\\2192"; position: absolute; right: -.52rem; z-index: 2;
        color: #176b87; font-size: 1.05rem; font-weight: 900; background: var(--canvas); }
    .research-note { margin: .15rem 0 .55rem; padding: .36rem .65rem; border-left: 3px solid #d0a23a;
        background: #fff9e9; color: #3d4a56 !important; font-size: .75rem; line-height: 1.3; }
    .section-kicker { color: #42576b; font-size: .74rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin: .15rem 0 .3rem; }
    .demo-controls { padding: .65rem .8rem .25rem; background: #ffffff; border: 1px solid var(--line); border-radius: 11px; }
    .image-frame { padding: .4rem; background: #102a43; border-radius: 11px; }
    .result-card { height: 100%; min-height: 7.4rem; padding: .8rem .9rem; border: 1px solid var(--line);
        border-top: 5px solid var(--blue); border-radius: 10px; background: #ffffff; box-shadow: 0 2px 7px rgba(18,59,93,.08); }
    .result-card.conflicting { border-top-color: #b33d3d; }
    .result-card.irrelevant { border-top-color: #926313; }
    .result-card.insufficient { border-top-color: #6c4c8a; }
    .result-card.incomplete { border-top-color: #b87516; }
    .card-label { color: #42576b; font-size: .69rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
    .card-value { color: #102f4a; margin: .22rem 0 .12rem; font-size: 1.42rem; font-weight: 900; line-height: 1.1; }
    .card-value.action { font-size: 1.16rem; }
    .gate-reason { margin: .4rem 0 .5rem; padding: .48rem .65rem; border-radius: 7px; background: #e9f1f6;
        color: #183b56 !important; font-size: .83rem; font-weight: 600; }
    .report-card { min-height: 9.3rem; height: 100%; padding: .78rem .9rem; border: 1px solid var(--line);
        border-radius: 10px; background: #ffffff; box-shadow: 0 2px 7px rgba(18,59,93,.06); }
    .report-title { color: #123b5d; font-size: .78rem; font-weight: 900; letter-spacing: .045em; text-transform: uppercase; }
    .frozen-tag { display: inline-block; margin: .28rem 0 .4rem; padding: .14rem .42rem; border-radius: 20px;
        background: #edf3f7; color: #344f64; font-size: .64rem; font-weight: 800; }
    .report-body { color: #203449; font-size: .85rem; line-height: 1.4; }
    .report-meta { margin-top: .4rem; color: #526779; font-size: .68rem; }
    .versus { display: flex; height: 100%; align-items: center; justify-content: center; color: #526779; font-size: .72rem; font-weight: 900; }
    .secondary-note { color: #42576b !important; font-size: .76rem; }
    @media (max-width: 900px) {
        .block-container { padding: .65rem .65rem 1rem; }
        .flow { grid-template-columns: repeat(3, 1fr); }
        .flow-step:not(:last-child)::after { content: ""; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title-row"><h1>Evidence-State Radiology</h1>'
    '<span class="title-tag">Phase 28A | Live workflow demonstrator</span></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="how-title">How the system works</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="flow">'
    '<div class="flow-step">IMAGE +<br>CLINICAL CONTEXT</div>'
    '<div class="flow-step">EVIDENCE-STATE<br>ANALYZER</div>'
    '<div class="flow-step">EVIDENCE<br>STATE</div>'
    '<div class="flow-step">RELIABILITY<br>GATE</div>'
    '<div class="flow-step">ACTION</div>'
    '<div class="flow-step">FROZEN MAIRA-2<br>OUTPUTS</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="research-note"><b>Research prototype:</b> the analyzer uses explicit reviewer inputs; '
    'it does not infer clinical relevance or image findings from raw text or pixels. Reports below are '
    '<b>Frozen Phase 26D result</b> records, not live inference.</div>',
    unsafe_allow_html=True,
)

try:
    records = load_generation_records()
    master = load_master_results()
except DemoDataError as exc:
    st.error(str(exc))
    st.stop()

conditions_in_data = [condition for condition in CONDITIONS if condition in set(records["condition"])]
sample_ids = sorted(records["sample_id"].astype(str).unique())
if not sample_ids or not conditions_in_data:
    st.error("No usable frozen Phase 26D study-condition records are available.")
    st.stop()

initial_condition = "sufficient" if "sufficient" in conditions_in_data else conditions_in_data[0]
if "selected_sample" not in st.session_state or st.session_state.selected_sample not in sample_ids:
    st.session_state.selected_sample = find_quick_demo_sample(records, initial_condition)
if "selected_condition" not in st.session_state or st.session_state.selected_condition not in conditions_in_data:
    st.session_state.selected_condition = initial_condition
if "clinical_context" not in st.session_state:
    st.session_state.clinical_context = get_study_context(
        records, st.session_state.selected_sample, st.session_state.selected_condition
    )
if "review_relevant" not in st.session_state:
    st.session_state.review_relevant = True
if "review_consistent" not in st.session_state:
    st.session_state.review_consistent = True
if "review_incomplete" not in st.session_state:
    st.session_state.review_incomplete = False
if "evidence_strength" not in st.session_state:
    st.session_state.evidence_strength = 1.0


def choose_quick_demo(condition: str) -> None:
    sample_id = find_quick_demo_sample(records, condition)
    st.session_state.selected_sample = sample_id
    st.session_state.selected_condition = condition
    st.session_state.clinical_context = get_study_context(records, sample_id, condition)
    # Seed transparent reviewer controls for the walkthrough; these are not model predictions.
    st.session_state.review_relevant = condition != "irrelevant"
    st.session_state.review_consistent = condition != "conflicting"
    st.session_state.review_incomplete = False
    st.session_state.evidence_strength = 1.0
    # Demonstrate insufficient evidence through the analyzer's existing no-context rule.
    if condition == "insufficient":
        st.session_state.clinical_context = ""


def load_selected_context() -> None:
    st.session_state.clinical_context = get_study_context(
        records, st.session_state.selected_sample, st.session_state.selected_condition
    )
    st.session_state.review_incomplete = False


st.markdown('<div class="section-kicker">Quick demo | choose a walkthrough</div>', unsafe_allow_html=True)
quick_cols = st.columns(4, gap="small")
for col, condition, label in zip(
    quick_cols,
    ("sufficient", "conflicting", "insufficient", "irrelevant"),
    ("Sufficient", "Conflicting", "Insufficient", "Irrelevant"),
):
    if condition in conditions_in_data and col.button(label, width="stretch"):
        choose_quick_demo(condition)
        st.rerun()
st.caption("Quick demos select a saved study and initialize the visible reviewer inputs; they are illustrative scenarios, not model predictions.")

selection_cols = st.columns([1.15, 1], gap="small")
with selection_cols[0]:
    st.selectbox(
        "Study",
        sample_ids,
        key="selected_sample",
        on_change=load_selected_context,
        format_func=lambda value: f"{value} | UID {int(records.loc[records['sample_id'].astype(str).eq(value), 'uid'].iloc[0])}",
    )
with selection_cols[1]:
    st.selectbox(
        "Frozen benchmark condition (metadata)",
        conditions_in_data,
        key="selected_condition",
        on_change=load_selected_context,
    )
st.caption("The benchmark condition is displayed for context and is not passed to the analyzer.")

selected = get_frozen_pair(records, st.session_state.selected_sample, st.session_state.selected_condition)
image_row = selected.get(0.0)
if image_row is None:
    image_row = selected.get(0.25)
image_path = resolve_image_path(image_row.get("frontal_image") if image_row is not None else "")
image_available = image_path is not None

work_cols = st.columns([1.13, 1], gap="medium")
with work_cols[0]:
    st.markdown('<div class="section-kicker">Image evidence | saved study image</div>', unsafe_allow_html=True)
    if image_path:
        st.image(str(image_path), caption=f"Selected study image | {image_path.name}", width=650)
    else:
        st.info("The exact image path recorded for this study is unavailable locally. No replacement image is used.")

with work_cols[1]:
    st.markdown('<div class="section-kicker">Clinical context | editable reviewer input</div>', unsafe_allow_html=True)
    st.text_area(
        "Clinical context",
        key="clinical_context",
        height=118,
        help="Edit this text to explore the structural completeness parser. Semantic relevance and consistency are reviewer-provided below.",
    )
    with st.expander("Frozen Phase 26D context (source record)", expanded=False):
        frozen_context = get_study_context(
            records, st.session_state.selected_sample, st.session_state.selected_condition
        )
        st.write(frozen_context or "No context stored for this selection.")
    signal_cols = st.columns(2, gap="small")
    with signal_cols[0]:
        st.checkbox("Context is relevant", key="review_relevant")
        st.checkbox("Reviewer marks context incomplete", key="review_incomplete")
    with signal_cols[1]:
        st.checkbox("Context is consistent", key="review_consistent")
        st.slider("Reviewer-assigned evidence strength", 0.0, 1.0, step=0.05, key="evidence_strength")
    parser_result = parse_context_completeness(st.session_state.clinical_context)
    parser_caption = (
        f"Structural parser: {'complete' if parser_result.is_complete else 'incomplete'} | "
        f"rule: {parser_result.rule_triggered}"
    )
    st.caption(parser_caption)
    assessment = EvidenceAssessment(
        image_available=image_available,
        context_available=bool(st.session_state.clinical_context.strip()),
        context_relevant=st.session_state.review_relevant,
        context_consistent=st.session_state.review_consistent,
        evidence_strength=st.session_state.evidence_strength,
        context_complete=parser_result.is_complete and not st.session_state.review_incomplete,
    )
    state = EvidenceStateAnalyzer().classify(assessment)
    decision = ReliabilityGate().decide(state)
    state_class = state.value.lower().replace("_", "-")
    state_label = state.value.replace("_", " ").upper()
    action_label = decision.action.replace("_", " ").upper()

    st.markdown('<div class="section-kicker">Live analyzer and reliability gate</div>', unsafe_allow_html=True)
    result_cols = st.columns([1, 1.35], gap="small")
    with result_cols[0]:
        st.markdown(
            f'<div class="result-card {html.escape(state_class)}">'
            f'<div class="card-label">Evidence State</div><div class="card-value">{html.escape(state_label)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with result_cols[1]:
        st.markdown(
            f'<div class="result-card {html.escape(state_class)}">'
            f'<div class="card-label">Gate Action</div><div class="card-value action">{html.escape(action_label)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="gate-reason"><b>Gate reason:</b> {html.escape(decision.reason)}</div>', unsafe_allow_html=True)
    st.caption("Existing analyzer and gate are called directly; the saved condition remains metadata.")

st.markdown('<div class="section-kicker">Frozen report comparison | precomputed outputs</div>', unsafe_allow_html=True)
report_cols = st.columns([1, .16, 1], gap="small")
report_specs = (
    (report_cols[0], 0.0, "BASELINE | lambda = 0.00"),
    (report_cols[2], 0.25, "EVIDENCE-AWARE | lambda = 0.25 (FROZEN)"),
)
for col, lambda_value, label in report_specs:
    with col:
        row = selected.get(lambda_value)
        if row is None:
            status_line = "No saved report exists for this study and condition."
            report_body = ""
            report_meta = ""
        else:
            status = safe_text(row.get("status"))
            report_body = safe_text(row.get("report"))
            if status.lower() != "success" or not report_body:
                status_line = f"Saved generation status: {status or 'unknown'}; report is unavailable."
                report_body = ""
            else:
                status_line = ""
            report_meta = (
                f"Saved output | {safe_text(row.get('num_tokens')) or '-'} tokens | "
                f"{safe_text(row.get('runtime_seconds')) or '-'} seconds"
            ) if report_body else ""
        st.markdown(
            '<div class="report-card">'
            f'<div class="report-title">{html.escape(label)}</div>'
            '<div class="frozen-tag">Frozen Phase 26D result</div>'
            f'<div class="report-body">{html.escape(report_body or status_line)}</div>'
            f'<div class="report-meta">{html.escape(report_meta)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
with report_cols[1]:
    st.markdown('<div class="versus">VS</div>', unsafe_allow_html=True)

st.markdown('<div class="section-kicker" style="margin-top: .8rem">Frozen Phase 26D quantitative snapshot</div>', unsafe_allow_html=True)
snapshot_metrics = ("Unsupported claim rate", "RadGraph F1", "Custom CheXpert-style F1")
snapshot = master.loc[
    master["metric"].isin(snapshot_metrics),
    ["metric", "lambda_0", "lambda_025", "interpretation"],
]
if snapshot.empty:
    st.warning("The expected metrics were not found in the frozen master table.")
else:
    st.dataframe(
        snapshot.rename(
            columns={
                "metric": "Metric",
                "lambda_0": "Baseline | lambda = 0.00",
                "lambda_025": "Evidence-aware | lambda = 0.25",
                "interpretation": "Recorded interpretation",
            }
        ),
        hide_index=True,
        width="stretch",
    )
st.caption(
    "Scientific note: held-out evaluation did not demonstrate statistically significant overall improvement. "
    "Results vary across metrics and evidence conditions; lambda = 0.25 remains frozen."
)
