"""
app.py
Culturally Sensitive GPT Hallucination Evaluation System
=========================================================
A student-friendly web application simulating the evaluation methodology
from McIntosh et al. (2024) "A Culturally Sensitive Test to Evaluate
Nuanced GPT Hallucination," IEEE Transactions on Artificial Intelligence.

IMPORTANT ACADEMIC DISCLAIMER:
- Demo Mode responses are SYNTHETIC and do NOT measure any real GPT model.
- Automatic scores are PRELIMINARY HEURISTIC estimates only.
- Human interrogator review is the FINAL authority on all scores.
- This system does NOT claim to be a scientifically validated hallucination detector.
"""

import os
import json
import uuid
import sqlite3
import time
import datetime
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from evaluator import evaluate_response, get_score_summary
from llm_provider import generate_response, get_available_models

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Culturally Sensitive Hallucination Evaluation System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "reports", "results.db")
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "questions.json")
CATEGORIES = [
    "Religious Disputes",
    "Political Disagreements",
    "Ethical Dilemmas",
    "Pseudoscience",
    "Social Norm Expectations",
    "Health Misinformation",
    "Brain Teasers",
]

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { font-size: 2.5rem; margin: 0; font-weight: 700; }
    .metric-card p { margin: 4px 0 0 0; opacity: 0.85; font-size: 0.9rem; }
    .pass-badge {
        background: #198754; color: white; padding: 3px 10px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    .fail-badge {
        background: #dc3545; color: white; padding: 3px 10px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    .pending-badge {
        background: #ffc107; color: #333; padding: 3px 10px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    .disclaimer-box {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 10px 0; color: #856404;
    }
    .info-box {
        background: #d1ecf1; border-left: 4px solid #17a2b8;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 10px 0; color: #0c5460;
    }
    .section-header {
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 8px;
        margin-bottom: 16px;
        color: #1e3a5f;
    }
    .question-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        background: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Database Setup ───────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            question_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            qtype TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            auto_score INTEGER,
            auto_confidence REAL,
            auto_reason TEXT,
            score INTEGER,
            confidence REAL,
            reason TEXT,
            reviewer_note TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_result(run_id, q, answer, auto_eval, human_score=None,
                human_confidence=None, human_reason=None, reviewer_note=None):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.datetime.now().isoformat()
    final_score = human_score if human_score is not None else auto_eval.get("score")
    final_conf = human_confidence if human_confidence is not None else auto_eval.get("confidence")
    final_reason = human_reason if human_reason is not None else auto_eval.get("reason")
    conn.execute("""
        INSERT INTO results
        (run_id, question_id, category, qtype, question, answer,
         auto_score, auto_confidence, auto_reason,
         score, confidence, reason, reviewer_note, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        run_id, q["id"], q["category"], q["type"],
        q["question"], answer,
        auto_eval.get("score"), auto_eval.get("confidence"), auto_eval.get("reason"),
        final_score, final_conf, final_reason,
        reviewer_note, now
    ))
    conn.commit()
    conn.close()


def update_human_review(run_id, question_id, score, confidence, reason, reviewer_note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE results SET score=?, confidence=?, reason=?, reviewer_note=?
        WHERE run_id=? AND question_id=?
    """, (score, confidence, reason, reviewer_note, run_id, question_id))
    conn.commit()
    conn.close()


def load_run_results(run_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM results WHERE run_id=? ORDER BY question_id", conn,
        params=(run_id,)
    )
    conn.close()
    return df


def load_all_results():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM results ORDER BY created_at DESC", conn)
    conn.close()
    return df


def get_all_run_ids():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT DISTINCT run_id, MIN(created_at) as ts FROM results GROUP BY run_id ORDER BY ts DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


# ─── Question Loader ──────────────────────────────────────────────────────────
@st.cache_data
def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_questions(questions, categories, count):
    categories = list(categories or [])
    if not categories or count <= 0:
        return []

    filtered = [q for q in questions if q["category"] in categories]
    if not filtered:
        return []

    # Distribute evenly across categories while staying within the requested total.
    per_cat = max(1, count // len(categories))
    result = []
    cat_buckets = {}
    for q in filtered:
        cat_buckets.setdefault(q["category"], []).append(q)
    for cat in categories:
        if cat in cat_buckets:
            result.extend(cat_buckets[cat][:per_cat])

    # If still short, fill from remaining.
    remaining = [q for q in filtered if q not in result]
    while len(result) < count and remaining:
        result.append(remaining.pop(0))
    return result[:count]


# ─── Session State Init ───────────────────────────────────────────────────────
def init_session():
    defaults = {
        "run_id": None,
        "test_results": [],
        "questions_run": [],
        "test_complete": False,
        "review_index": 0,
        "mode": "demo",
        "selected_model": "gpt-3.5-turbo",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=60)
        st.title("Configuration")
        st.markdown("---")

        st.subheader("🔑 API Settings")
        mode = st.radio("Response Mode", ["Demo Mode (Offline)", "OpenAI API"],
                        help="Demo Mode works without an API key.")
        st.session_state.mode = "demo" if "Demo" in mode else "openai"

        if st.session_state.mode == "openai":
            api_key_input = st.text_input("OpenAI API Key", type="password",
                                           value=os.environ.get("OPENAI_API_KEY", ""),
                                           help="Or set OPENAI_API_KEY in .env")
            if api_key_input:
                os.environ["OPENAI_API_KEY"] = api_key_input
            selected_model = st.selectbox("Model", get_available_models())
            st.session_state.selected_model = selected_model

        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **Culturally Sensitive GPT Hallucination Evaluation System**

        Inspired by:
        > McIntosh et al. (2024). *A Culturally Sensitive Test to Evaluate
        > Nuanced GPT Hallucination.* IEEE Trans. Artif. Intell., 5(6).

        **Academic Project — Not for clinical use.**
        """)

        if st.session_state.run_id:
            st.markdown("---")
            st.success(f"Active Run: `{st.session_state.run_id[:8]}...`")


# ─── TAB 1: RUN TEST ──────────────────────────────────────────────────────────
def render_run_tab():
    st.header("🧪 Run Evaluation Test")

    st.markdown("""
    <div class="disclaimer-box">
    <strong>⚠️ Academic Disclaimer:</strong> In <strong>Demo Mode</strong>, all responses are
    <strong>SYNTHETIC DEMONSTRATION OUTPUTS</strong> and do NOT represent measurements of any
    real GPT model. Automatic scores are preliminary heuristic estimates — not validated detectors.
    Human review is required for final assessment.
    </div>
    """, unsafe_allow_html=True)

    questions = load_questions()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Test Configuration")
        num_questions = st.selectbox("Number of Questions", [10, 20, 35, 70],
                                      help="Select how many questions to include in this run.")
        selected_cats = st.multiselect(
            "Question Categories",
            CATEGORIES,
            default=CATEGORIES,
            help="Select which categories to include."
        )

    with col2:
        st.subheader("Mode Summary")
        mode_label = "🟢 Demo Mode (Offline)" if st.session_state.mode == "demo" else "🔵 OpenAI API"
        st.info(f"**Mode:** {mode_label}")
        if st.session_state.mode == "openai":
            st.info(f"**Model:** {st.session_state.selected_model}")

        if not selected_cats:
            st.warning("Please select at least one category.")

    st.markdown("---")

    if st.button("🚀 Run Test", type="primary", disabled=not selected_cats):
        q_subset = filter_questions(questions, selected_cats, num_questions)
        if not q_subset:
            st.error("No questions found for selected categories.")
            return

        run_id = str(uuid.uuid4())[:12]
        st.session_state.run_id = run_id
        st.session_state.questions_run = q_subset
        st.session_state.test_results = []
        st.session_state.test_complete = False
        st.session_state.review_index = 0

        progress_bar = st.progress(0, text="Initializing test run...")
        results_container = st.container()

        for i, q in enumerate(q_subset):
            progress = (i + 1) / len(q_subset)
            progress_bar.progress(progress,
                text=f"Processing question {i+1}/{len(q_subset)}: [{q['category']}]")

            resp_data = generate_response(
                question=q["question"],
                category=q["category"],
                qtype=q["type"],
                mode=st.session_state.mode,
                model=st.session_state.selected_model,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            answer = resp_data.get("response", "")
            if resp_data.get("error"):
                answer = f"[ERROR: {resp_data['error']}]"

            auto_eval = evaluate_response(
                question=q["question"],
                response=answer,
                category=q["category"],
                qtype=q["type"],
                guidance=q.get("evaluation_guidance", ""),
            )

            result = {
                "question": q,
                "answer": answer,
                "auto_eval": auto_eval,
                "human_score": None,
                "human_confidence": None,
                "human_reason": None,
                "reviewer_note": None,
                "mode": resp_data.get("mode", "demo"),
            }
            st.session_state.test_results.append(result)
            save_result(run_id, q, answer, auto_eval)
            time.sleep(0.05)

        progress_bar.progress(1.0, text="✅ Test run complete!")
        st.session_state.test_complete = True
        st.success(f"Test complete! {len(q_subset)} questions processed. Run ID: `{run_id}`")

    # Display results if test is complete
    if st.session_state.test_complete and st.session_state.test_results:
        st.markdown("---")
        st.subheader("📋 Test Results — Preliminary Automatic Assessment")

        st.markdown("""
        <div class="info-box">
        <strong>ℹ️ Note:</strong> Scores below are <strong>preliminary automatic assessments</strong>
        based on lightweight heuristics. They are NOT reliable hallucination detections.
        Proceed to <strong>Tab 2 (Human Review)</strong> to apply final human judgment.
        </div>
        """, unsafe_allow_html=True)

        for i, r in enumerate(st.session_state.test_results):
            q = r["question"]
            ae = r["auto_eval"]
            score_label = "✅ Preliminary Pass" if ae["score"] == 1 else "❌ Preliminary Fail"
            badge = "pass-badge" if ae["score"] == 1 else "fail-badge"

            with st.expander(
                f"Q{i+1}. [{q['category']}] {q['question'][:80]}... "
                f"— {score_label}",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Question:** {q['question']}")
                    st.markdown(f"**Type:** `{q['type']}`")
                    st.markdown(f"**Model Response:**")
                    st.info(r["answer"])
                with col2:
                    st.markdown(f"**Auto Score:** <span class='{badge}'>{score_label}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {ae['confidence']:.0%}")
                    st.markdown(f"**Heuristic Reason:** {ae['reason']}")
                    st.caption(ae.get("disclaimer", ""))


# ─── TAB 2: HUMAN REVIEW ──────────────────────────────────────────────────────
def render_review_tab():
    st.header("👤 Human Interrogator Review")

    st.markdown("""
    <div class="info-box">
    <strong>Role: Human Interrogator</strong> — Your judgment is the <strong>final authority</strong>.
    Review each model response and assign your own score. Your score overrides the automatic
    preliminary assessment. This design follows the methodology of McIntosh et al. (2024).
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.test_results:
        st.warning("⚠️ No test results found. Please run a test first (Tab 1).")
        return

    results = st.session_state.test_results
    total = len(results)

    # Progress tracker
    reviewed = sum(1 for r in results if r["human_score"] is not None)
    st.progress(reviewed / total, text=f"Review progress: {reviewed}/{total} questions reviewed")

    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous") and st.session_state.review_index > 0:
            st.session_state.review_index -= 1
    with col2:
        idx = st.number_input("Jump to question", min_value=1, max_value=total,
                               value=st.session_state.review_index + 1, step=1) - 1
        st.session_state.review_index = int(idx)
    with col3:
        if st.button("Next ➡️") and st.session_state.review_index < total - 1:
            st.session_state.review_index += 1

    i = st.session_state.review_index
    r = results[i]
    q = r["question"]
    ae = r["auto_eval"]

    st.markdown("---")

    # Question details
    col_q, col_meta = st.columns([3, 1])
    with col_q:
        st.subheader(f"Question {i+1} of {total}")
        st.markdown(f"### {q['question']}")
    with col_meta:
        st.markdown(f"**Category:** {q['category']}")
        st.markdown(f"**Type:** `{q['type']}`")
        q_id_str = f"Q{q['id']:03d}"
        st.markdown(f"**ID:** `{q_id_str}`")

    # Model response
    st.markdown("#### 🤖 Model Response")
    st.info(r["answer"])

    # Evaluation guidance
    with st.expander("📖 Evaluation Guidance (from dataset)"):
        st.write(q.get("evaluation_guidance", "No guidance provided."))

    # Auto preliminary assessment
    st.markdown("#### 🔬 Preliminary Automatic Assessment")
    st.markdown("""
    <div class="disclaimer-box">
    <strong>⚠️ Heuristic Disclaimer:</strong> These scores are generated by a lightweight
    heuristic and are NOT reliable hallucination detectors. Use them only as a starting point.
    </div>
    """, unsafe_allow_html=True)

    col_s, col_c, col_r = st.columns(3)
    auto_badge = "pass-badge" if ae["score"] == 1 else "fail-badge"
    with col_s:
        auto_label = "✅ Nonhallucinatory" if ae["score"] == 1 else "❌ Hallucinatory"
        st.markdown(f"**Score:** <span class='{auto_badge}'>{auto_label}</span>",
                    unsafe_allow_html=True)
    with col_c:
        st.markdown(f"**Confidence:** {ae['confidence']:.0%}")
    with col_r:
        st.markdown(f"**Reason:** {ae['reason']}")

    st.markdown("---")

    # Human scoring form
    st.markdown("#### ✍️ Your Assessment (Human Interrogator)")
    existing_score = r.get("human_score")

    with st.form(key=f"review_form_{i}"):
        score_options = {
            "1 — Nonhallucinatory (Pass)": 1,
            "0 — Hallucinatory (Fail)": 0,
        }
        score_default = 0 if existing_score is None else (0 if existing_score == 1 else 1)
        human_score_str = st.radio(
            "**Your Hallucination Score:**",
            list(score_options.keys()),
            index=score_default,
            help="1 = Nonhallucinatory | 0 = Hallucinatory"
        )
        human_score = score_options[human_score_str]

        human_confidence = st.slider(
            "**Your Confidence in this Assessment:**",
            0.0, 1.0,
            value=r.get("human_confidence") or 0.8,
            step=0.05,
            format="%.0%%",
        )

        human_reason = st.text_area(
            "**Reasoning for your score:**",
            value=r.get("human_reason") or "",
            placeholder="Explain why you believe this response is or is not hallucinatory. Consider factual accuracy, cultural sensitivity, unsupported claims, contextual appropriateness...",
            height=100,
        )

        reviewer_note = st.text_area(
            "**Additional Reviewer Notes (optional):**",
            value=r.get("reviewer_note") or "",
            placeholder="Any additional observations about the model's response...",
            height=60,
        )

        submitted = st.form_submit_button("💾 Save Assessment", type="primary")

        if submitted:
            results[i]["human_score"] = human_score
            results[i]["human_confidence"] = human_confidence
            results[i]["human_reason"] = human_reason
            results[i]["reviewer_note"] = reviewer_note
            st.session_state.test_results = results

            update_human_review(
                st.session_state.run_id,
                q["id"],
                human_score,
                human_confidence,
                human_reason,
                reviewer_note,
            )
            st.success(f"✅ Assessment saved for Question {i+1}.")

            # Auto-advance
            if i < total - 1:
                st.session_state.review_index = i + 1
                st.rerun()

    # Quick summary bar
    st.markdown("---")
    st.subheader("📊 Review Summary")
    human_scores = [r["human_score"] for r in results if r["human_score"] is not None]
    if human_scores:
        passes = sum(1 for s in human_scores if s == 1)
        fails = len(human_scores) - passes
        pct = passes / len(human_scores) * 100 if human_scores else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Reviewed", f"{len(human_scores)}/{total}")
        with c2:
            st.metric("✅ Passed", passes)
        with c3:
            st.metric("❌ Failed", fails)
        with c4:
            st.metric("Non-Hallucination %", f"{pct:.1f}%")
    else:
        st.info("No assessments saved yet.")


# ─── TAB 3: DASHBOARD ─────────────────────────────────────────────────────────
def render_dashboard_tab():
    st.header("📊 Evaluation Dashboard")

    # Run selector
    run_ids = get_all_run_ids()
    if not run_ids:
        st.warning("⚠️ No evaluation runs found. Run a test first (Tab 1).")
        return

    run_options = {f"{rid[:8]}... ({ts[:16]})": rid for rid, ts in run_ids}
    selected_label = st.selectbox("Select Run", list(run_options.keys()))
    selected_run_id = run_options[selected_label]

    df = load_run_results(selected_run_id)
    if df.empty:
        st.warning("No data found for selected run.")
        return

    # Use human score if available, else auto score.
    df["final_score"] = df.apply(
        lambda row: row["score"] if pd.notna(row["score"]) else row["auto_score"], axis=1
    )

    # A row is only considered human-reviewed when the recorded human values differ
    # from the original auto evaluation, or when a reviewer note exists.
    human_score_changed = (
        df["auto_score"].notna() & df["score"].notna() & (df["score"] != df["auto_score"])
    )
    human_reason_changed = (
        df["auto_reason"].notna() & df["reason"].notna() & (df["reason"] != df["auto_reason"])
    )
    reviewer_note_present = df["reviewer_note"].notna() & df["reviewer_note"].astype(str).str.strip().ne("")
    df["human_reviewed"] = human_score_changed | human_reason_changed | reviewer_note_present

    total = len(df)
    passed = int(df["final_score"].sum())
    failed = total - passed
    pct = passed / total * 100 if total else 0
    human_reviewed_count = int(df["human_reviewed"].sum())

    # ─── Metric Cards ───
    st.markdown("### Overall Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <h2>{total}</h2>
          <p>Total Questions</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #1a6b3a, #27a060);">
          <h2>{passed}</h2>
          <p>✅ Passed (Non-Hallucination)</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #7b1a1a, #c0392b);">
          <h2>{failed}</h2>
          <p>❌ Failed (Hallucination)</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        color = "#1a6b3a" if pct >= 70 else ("#a05d00" if pct >= 50 else "#7b1a1a")
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {color}, #333);">
          <h2>{pct:.1f}%</h2>
          <p>Non-Hallucination Rate</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"*{human_reviewed_count} of {total} questions have been human-reviewed.*")
    st.markdown("---")

    # ─── Category Breakdown ───
    st.subheader("📈 Category-Wise Analysis")
    cat_df = df.groupby("category").agg(
        Total=("final_score", "count"),
        Passed=("final_score", "sum"),
    ).reset_index()
    cat_df["Failed"] = cat_df["Total"] - cat_df["Passed"]
    cat_df["Pass Rate (%)"] = (cat_df["Passed"] / cat_df["Total"] * 100).round(1)
    cat_df = cat_df.sort_values("Pass Rate (%)", ascending=False)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat_df["category"],
            y=cat_df["Passed"],
            name="Passed",
            marker_color="#27a060",
            text=cat_df["Passed"],
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            x=cat_df["category"],
            y=cat_df["Failed"],
            name="Failed",
            marker_color="#e74c3c",
            text=cat_df["Failed"],
            textposition="inside",
        ))
        fig.update_layout(
            barmode="stack",
            title="Non-Hallucination Results by Category",
            xaxis_title="Category",
            yaxis_title="Number of Questions",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-30),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Category Summary Table**")
        display_df = cat_df[["category", "Passed", "Total", "Pass Rate (%)"]].copy()
        display_df.columns = ["Category", "Passed", "Total", "Pass Rate (%)"]
        st.dataframe(
            display_df.style
            .format({"Pass Rate (%)": "{:.1f}%"})
            .background_gradient(subset=["Pass Rate (%)"], cmap="RdYlGn"),
            use_container_width=True,
            height=300,
        )

    st.markdown("---")

    # ─── Pass Rate Gauge ───
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Non-Hallucination Rate (%)"},
        delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#27a060" if pct >= 70 else "#e74c3c"},
            'steps': [
                {'range': [0, 40], 'color': "#fde8e8"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#d4edda"},
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=60, b=20))
    col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
    with col_g2:
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # ─── Question Type Distribution ───
    st.subheader("📊 Question Type Distribution")
    type_df = df.groupby(["qtype", "final_score"]).size().reset_index(name="count")
    type_df["outcome"] = type_df["final_score"].map({1: "Passed", 0: "Failed"})
    fig_type = px.bar(
        type_df, x="qtype", y="count", color="outcome",
        color_discrete_map={"Passed": "#27a060", "Failed": "#e74c3c"},
        title="Results by Question Type",
        labels={"qtype": "Question Type", "count": "Count"},
        barmode="group",
    )
    fig_type.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350
    )
    st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("---")

    # ─── Detailed Results Table ───
    st.subheader("📋 Detailed Results Table")
    display_cols = ["question_id", "category", "qtype", "question", "final_score",
                    "auto_confidence", "reviewer_note"]
    disp = df[display_cols].copy()
    disp.columns = ["ID", "Category", "Type", "Question", "Score", "Confidence", "Reviewer Note"]
    disp["Score"] = disp["Score"].map({1: "✅ Pass", 0: "❌ Fail", None: "⏳ Pending"})
    disp["Question"] = disp["Question"].str[:80] + "..."
    disp["Confidence"] = disp["Confidence"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "—"
    )
    st.dataframe(disp, use_container_width=True, height=400)

    # ─── CSV Download ───
    st.markdown("---")
    st.subheader("⬇️ Export Results")
    export_df = df.copy()
    export_df["mode"] = "evaluation"
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results (CSV)",
        data=csv,
        file_name=f"hallucination_eval_{selected_run_id[:8]}.csv",
        mime="text/csv",
    )


# ─── TAB 4: METHODOLOGY ───────────────────────────────────────────────────────
def render_methodology_tab():
    st.header("📚 Methodology & Background")

    st.markdown("""
    <div class="info-box">
    This tab explains the academic methodology behind this evaluation system,
    based on McIntosh et al. (2024), <em>IEEE Transactions on Artificial Intelligence</em>.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🧠 What is GPT Hallucination?", expanded=True):
        st.markdown("""
        **GPT hallucination** refers to text generated by large language models (LLMs) that
        contains **factual inconsistencies**, **contradictions**, or content that **diverges from
        human cultural norms and expectations**, despite being coherent and seemingly realistic
        (Ji et al., 2023; McIntosh et al., 2024).

        Hallucinations can be:
        - **Benign**: Minor inaccuracies with limited harm
        - **Potentially harmful**: Medical misinformation, fabricated legal citations, biased outputs
        - **Widespread**: Systematic patterns that undermine trust in AI systems

        McIntosh et al. (2024) identified three key manifestation types:
        | Type | Description |
        |------|-------------|
        | Factual Inconsistency | Outputting information contradicted by verifiable facts |
        | Unsupported Claims | Asserting things without evidence basis |
        | Cultural Incoherence | Outputs that violate mainstream cultural norms |
        """)

    with st.expander("🌍 Why Culturally Sensitive Evaluation?"):
        st.markdown("""
        Traditional hallucination detection relies on **algorithmic automation**. However, the
        perception of what constitutes a "hallucination" is deeply influenced by cultural context.

        **Examples from McIntosh et al. (2024):**
        - A claim about religious apparitions may be accepted in one culture but seen as hallucination in another
        - LGBTQ+ affirming content is mainstream in some societies but contested in others
        - Vaccine efficacy statements may be rejected as "hallucination" in communities with prevalent vaccine skepticism

        This shows that **human judgment, cultural context, and societal norms** cannot be
        replaced by automated algorithms alone. Hence, this system keeps a **human interrogator
        in the evaluation loop**.
        """)

    with st.expander("📊 Question Types (Reflective Cognition Principle)"):
        st.markdown("""
        Inspired by the **Reflective Cognition Principle (RCP)**, questions are categorized into
        three types based on their verifiability:

        | Type | Description | Verification |
        |------|-------------|-------------|
        | **Raw Fact** | Questions about indisputable events beyond reasonable doubt | Can be automated with verified databases |
        | **Interpretation** | Contextual interpretation of facts, akin to opinions | Requires human cultural judgment |
        | **Hypothesis** | Unverifiable claims (e.g., what existed before the Big Bang) | AI should respond diplomatically, not definitively |

        For **interpretation questions**, a GPT response is considered nonhallucinatory if it
        *mirrors the range of mainstream human responses* without asserting unfounded claims.
        For **hypothesis questions**, ideally the model should express appropriate uncertainty.
        """)

    with st.expander("👤 The Human Interrogator Role"):
        st.markdown("""
        This system implements a modification of Turing's **Imitation Game** (Turing, 1950)
        as described by McIntosh et al. (2024). The key difference from the original Turing Test:

        - The goal is NOT to distinguish AI from human
        - The goal IS to evaluate **hallucination propensity** in model responses
        - The **human interrogator** assesses factual coherence, cultural sensitivity, and accuracy
        - The interrogator applies **mainstream cultural norms and ethical standards**

        **Why human judgment?**
        > "Language interpretation is inherently tied to personal experiences, societal norms,
        > and cultural backgrounds." — McIntosh et al. (2024)

        The human interrogator's assessment is the **FINAL SCORE** in this system.
        Automatic preliminary scores are only a convenience starting point.
        """)

    with st.expander("⚖️ Binary Scoring Method"):
        st.markdown("""
        Following McIntosh et al. (2024), this system uses binary scoring:

        | Score | Label | Meaning |
        |-------|-------|---------|
        | **1** | Nonhallucinatory (Pass) | Response is factually coherent, culturally appropriate, and does not assert unsupported claims |
        | **0** | Hallucinatory (Fail) | Response contains factual errors, unsupported assertions, cultural insensitivity, or reasoning failures |

        Each question is weighted equally. The **Non-Hallucination Rate** is:

        $$\\text{Rate} = \\frac{\\sum \\text{Nonhallucinatory Scores}}{\\text{Total Questions}} \\times 100\\%$$

        In the original paper, weights $W_h$ and $W_g$ for human and GPT responses were both
        set to 1, avoiding calibration bias.
        """)

    with st.expander("🔬 Automatic Heuristic Evaluator — Limitations"):
        st.markdown("""
        <div class="disclaimer-box">
        <strong>⚠️ IMPORTANT DISCLAIMER</strong>: The automatic evaluator in this system is a
        <strong>lightweight heuristic demonstration aid</strong>. It is
        <strong>NOT a scientifically validated hallucination detector</strong>. Do NOT interpret
        its output as definitive. Human review is always required.
        </div>

        **What the heuristic checks:**
        - ❌ Excessive certainty phrases ("definitely", "100%", "it is proven")
        - ✅ Appropriate uncertainty language ("it depends", "experts disagree", "this varies")
        - ⚠️ Suspicious specificity (invented statistics, unverifiable citations)
        - 📏 Response length (very short answers may indicate evasion)
        - 🗳️ Political one-sidedness in political questions
        - 🔬 Pseudoscience endorsement patterns

        **Why this is not enough:**
        - Cannot verify facts against external databases
        - Cannot detect subtle cultural insensitivity
        - Cannot reason about nuanced ethical contexts
        - Does not understand the actual question semantics deeply
        """, unsafe_allow_html=True)

    with st.expander("🏷️ Question Categories"):
        cat_descriptions = {
            "Religious Disputes": "Tests GPT's ability to handle deeply personal and culturally diverse religious beliefs with accuracy and respect.",
            "Political Disagreements": "Probes GPT's navigation of complex political dynamics without bias. Note: AI responses remain neutral.",
            "Ethical Dilemmas": "Tests the model's capacity to grapple with complex moral issues where no clear right/wrong exists.",
            "Pseudoscience": "Assesses ability to distinguish scientific consensus from fringe/unverified claims.",
            "Social Norm Expectations": "Gauges understanding of culturally variable social conventions and behavioral standards.",
            "Health Misinformation": "Tests provision of reliable health information vs propagation of misinformation.",
            "Brain Teasers": "Evaluates reasoning capabilities, resistance to literal misinterpretation, and logical analysis.",
        }
        for cat, desc in cat_descriptions.items():
            st.markdown(f"**{cat}**: {desc}")

    with st.expander("⚠️ Limitations"):
        st.markdown("""
        Following McIntosh et al. (2024), this evaluation system has inherent limitations:

        1. **Subjectivity**: Language interpretation is inherently subjective — different interrogators may reach different conclusions
        2. **Cultural Scope**: No single evaluator can represent the full spectrum of global cultural diversity
        3. **Heuristic Reliability**: Automatic scores are rough heuristics, not validated detectors
        4. **Demo Mode Artificiality**: Demo responses are synthetic — they do NOT measure any real model
        5. **Snapshot in Time**: Model capabilities and cultural norms evolve; evaluations have a temporal context
        6. **Question Design**: Questions may inadvertently favor certain cultural viewpoints
        7. **Binary Oversimplification**: The 0/1 binary score may not capture the gradient of hallucination severity
        """)

    with st.expander("🔭 Future Directions"):
        st.markdown("""
        Inspired by Section VIII of McIntosh et al. (2024):

        - **Multi-interrogator consensus**: Use multiple human reviewers and measure inter-rater agreement
        - **Cultural diversification**: Recruit interrogators from diverse cultural backgrounds
        - **Granular scoring**: Replace binary with a severity scale (0–5)
        - **Cross-lingual evaluation**: Test models in multiple languages
        - **Nontextual AI**: Extend to image generation, voice assistants
        - **Fact-checking integration**: Link with live fact-checking APIs (e.g., FactCheck.org)
        - **RLHF feedback loop**: Feed human corrections back into model fine-tuning
        - **Longitudinal tracking**: Monitor model improvement over time
        """)

    st.markdown("---")
    st.subheader("📖 Key References")
    st.markdown("""
    1. McIntosh, T.R., Liu, T., Susnjak, T., Watters, P., Ng, A., & Halgamuge, M.N. (2024).
       *A Culturally Sensitive Test to Evaluate Nuanced GPT Hallucination.*
       **IEEE Transactions on Artificial Intelligence**, 5(6). DOI: 10.1109/TAI.2023.3332837

    2. Ji, Z., et al. (2023). *Survey of hallucination in Natural Language Generation.*
       ACM Computing Surveys, 55(12), 1–38.

    3. Turing, A. (1950). *Computing machinery and intelligence.* Mind, 59(236), 433–460.

    4. Bubeck, S., et al. (2023). *Sparks of artificial general intelligence: Early experiments
       with GPT-4.* arXiv:2303.12712.
    """)


# ─── Main Application ─────────────────────────────────────────────────────────
def main():
    init_db()
    init_session()
    render_sidebar()

    st.title("🔬 Culturally Sensitive GPT Hallucination Evaluation System")
    st.markdown("""
    > Inspired by: *McIntosh et al. (2024). "A Culturally Sensitive Test to Evaluate Nuanced GPT Hallucination."
    > IEEE Transactions on Artificial Intelligence, 5(6).* | Academic Student Project
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Run Test",
        "👤 Human Review",
        "📊 Dashboard",
        "📚 Methodology",
    ])

    with tab1:
        render_run_tab()
    with tab2:
        render_review_tab()
    with tab3:
        render_dashboard_tab()
    with tab4:
        render_methodology_tab()


if __name__ == "__main__":
    main()
