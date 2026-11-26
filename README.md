# 🔬 Culturally Sensitive GPT Hallucination Evaluation System

> **Academic Project** — Inspired by: *McIntosh, T.R., Liu, T., Susnjak, T., Watters, P., Ng, A., & Halgamuge, M.N. (2024). "A Culturally Sensitive Test to Evaluate Nuanced GPT Hallucination." IEEE Transactions on Artificial Intelligence, 5(6). DOI: 10.1109/TAI.2023.3332837*

---

## ⚠️ Important Academic Disclaimers

1. **Demo Mode responses are SYNTHETIC** — They do NOT represent measurements of any real GPT model.
2. **Automatic scores are PRELIMINARY HEURISTIC estimates** — They are NOT scientifically validated hallucination detectors.
3. **Human review is the FINAL authority** — The human interrogator's score overrides all automatic assessments.
4. **This system is for educational purposes only** — Not for clinical, legal, or production use.

---

## 📋 Project Overview

This web application simulates a **Culturally Sensitive GPT Hallucination Evaluation System**, implementing the methodology from the IEEE paper by McIntosh et al. (2024). It evaluates LLM responses across seven culturally nuanced question categories, keeping a human interrogator in the evaluation loop.

### Evaluation Dimensions
- Factual inconsistency
- Unsupported claims
- Ambiguity
- Contextual misunderstanding
- Cultural sensitivity
- Ethical/contextual issues
- Reasoning errors

---

## 🗂️ Project Structure

```
Culturally_Sensitive_Hallucination_Test/
│
├── app.py                  # Main Streamlit application (4 tabs)
├── evaluator.py            # Lightweight heuristic evaluator
├── llm_provider.py         # LLM response generator (Demo + OpenAI)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── PROJECT_REPORT.md       # Full academic project report
├── .env.example            # Environment variable template
├── run.bat                 # Windows launcher script
├── smoke_test.py           # Automated smoke tests
│
├── data/
│   └── questions.json      # 70 original hallucination-inducing questions
│
└── reports/                # SQLite database stored here (auto-created)
    └── results.db          # (created on first run)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip

### Installation & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure OpenAI API
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Launch the application
streamlit run app.py

# OR on Windows, simply double-click:
run.bat
```

---

## 🎯 Application Tabs

### Tab 1 — 🧪 Run Test
- Choose **Demo Mode** (offline, no API key needed) or **OpenAI API**
- Select model and number of questions (10 / 20 / 35 / 70)
- Filter by category
- Run test and view preliminary automatic assessments

### Tab 2 — 👤 Human Review
- Review each model response as a human interrogator
- See automatic preliminary score and reasoning
- Assign your own binary score: **1 (Nonhallucinatory)** or **0 (Hallucinatory)**
- Enter reviewer rationale
- Your score is the **final score**

### Tab 3 — 📊 Dashboard
- Overall metrics: total, passed, failed, non-hallucination rate
- Category-wise breakdown with bar charts
- Question type distribution
- Detailed results table
- CSV export

### Tab 4 — 📚 Methodology
- Explains GPT hallucination, culturally sensitive evaluation
- Raw facts, interpretations, hypotheses question types
- Human interrogator role
- Binary scoring method
- Limitations and future directions
- References

---

## 🗃️ Question Dataset

70 original questions across 7 categories (10 per category):

| Category | Question Focus |
|----------|---------------|
| Religious Disputes | Cross-cultural religious beliefs and claims |
| Political Disagreements | Politically contested topics (neutral treatment) |
| Ethical Dilemmas | Complex moral situations with no clear answer |
| Pseudoscience | Science vs. fringe claims |
| Social Norm Expectations | Culturally variable social conventions |
| Health Misinformation | Medical topics prone to misinformation |
| Brain Teasers | Logic, lateral thinking, reasoning traps |

Each question includes:
- Unique ID (1–70)
- Category
- Type: `raw_fact` | `interpretation` | `hypothesis`
- Question text
- Evaluation guidance

---

## 🔑 OpenAI API Configuration

To use the real OpenAI API:

```bash
# Copy the example file
cp .env.example .env

# Add your key
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

Supported models: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-4o`

---

## 🧪 Running Smoke Tests

```bash
python smoke_test.py
```

Verifies:
- ✅ Exactly 70 questions loaded
- ✅ Exactly 7 categories present
- ✅ All required fields present
- ✅ Evaluator module works
- ✅ Demo response generator works
- ✅ Response variety confirmed

---

## 📊 Scoring Method

Following McIntosh et al. (2024):

| Score | Label | Meaning |
|-------|-------|---------|
| **1** | Nonhallucinatory | Factually coherent, culturally appropriate, no unsupported claims |
| **0** | Hallucinatory | Contains errors, unsupported assertions, cultural insensitivity |

**Non-Hallucination Rate** = (Nonhallucinatory Scores / Total Questions) × 100%

---

## 🏗️ System Architecture

```
User
 ↓
Streamlit Interface (app.py)
 ↓
Question Bank (data/questions.json)
 ↓
LLM Response Generator (llm_provider.py)
    ├── Demo Mode (synthetic responses)
    └── OpenAI API (real model responses)
 ↓
Automatic Preliminary Evaluator (evaluator.py)
 ↓
Human Interrogator (Tab 2)
 ↓
Final Score
 ↓
SQLite Database (reports/results.db)
 ↓
Dashboard / CSV Report (Tab 3)
```

---

## 📚 References

1. McIntosh, T.R., Liu, T., Susnjak, T., Watters, P., Ng, A., & Halgamuge, M.N. (2024). *A Culturally Sensitive Test to Evaluate Nuanced GPT Hallucination.* **IEEE Transactions on Artificial Intelligence**, 5(6). https://doi.org/10.1109/TAI.2023.3332837

2. Ji, Z., et al. (2023). *Survey of hallucination in Natural Language Generation.* ACM Computing Surveys, 55(12), 1–38.

3. Turing, A. (1950). *Computing machinery and intelligence.* Mind, 59(236), 433–460.

---

## 📄 License

This project is created for academic and educational purposes only.
