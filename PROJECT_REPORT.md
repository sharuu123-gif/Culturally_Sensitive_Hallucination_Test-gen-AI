# PROJECT REPORT

## Culturally Sensitive GPT Hallucination Evaluation System

---

**Course/Subject:** Artificial Intelligence / Natural Language Processing  
**Academic Year:** 2024–2025  
**Project Type:** Software Implementation Project  
**Technology Stack:** Python, Streamlit, SQLite, Pandas, Plotly

---

## 1. Title

**Culturally Sensitive GPT Hallucination Evaluation System**  
*A Student Implementation Inspired by McIntosh et al. (2024)*

---

## 2. Abstract

Large Language Models (LLMs), particularly Generative Pretrained Transformers (GPTs), have transformed natural language processing but are susceptible to generating "hallucinations" — outputs that are factually inconsistent, culturally incoherent, or divergent from human expectations. This project implements a student-facing web application that simulates the evaluation methodology introduced by McIntosh et al. (2024) in their IEEE paper "A Culturally Sensitive Test to Evaluate Nuanced GPT Hallucination." The system evaluates LLM responses across seven culturally nuanced categories using a binary scoring scheme, with a mandatory human interrogator in the evaluation loop. The application supports both an offline demo mode (using synthetic responses) and integration with the OpenAI API. Results are stored in a local SQLite database and presented through an interactive dashboard built with Streamlit and Plotly.

> **Disclaimer:** All automatic scores are preliminary heuristic estimates and are NOT scientifically validated hallucination detections. Human review is the final authority. Demo mode responses are synthetic and do NOT represent measurements of any real GPT model.

---

## 3. Problem Statement

Modern GPT systems occasionally produce responses that contain:
- **Factual inconsistencies**: Statements contradicted by verifiable facts
- **Unsupported claims**: Assertions presented as fact without evidence
- **Cultural incoherence**: Outputs misaligned with mainstream cultural norms and values
- **Reasoning errors**: Flawed logical inferences or spatial/lateral reasoning failures

Current automated hallucination detection methods face fundamental limitations:
1. They cannot account for cultural subjectivity in what constitutes "truth"
2. They miss subtle contextual inaccuracies
3. They replace human judgment with algorithmic approximations

As McIntosh et al. (2024) argue: *"The evaluation of GPT hallucinations cannot rely solely on algorithmic methods."* A human interrogator must remain in the loop, especially for interpretation-type questions where cultural context shapes what is considered hallucinatory.

---

## 4. Objectives

1. **Implement** a working simulation of the Culturally Sensitive Test methodology from McIntosh et al. (2024)
2. **Create** a dataset of 70 original, hallucination-inducing questions across 7 culturally nuanced categories
3. **Build** a Streamlit web interface with four functional tabs: Run Test, Human Review, Dashboard, Methodology
4. **Integrate** both an offline demo mode and optional OpenAI API connectivity
5. **Implement** a lightweight heuristic evaluator as a demonstration aid (not a validated detector)
6. **Maintain** a human interrogator as the final scoring authority
7. **Store** all results in a SQLite database for analysis and CSV export
8. **Demonstrate** through charts and metrics how different question categories affect hallucination rates

---

## 5. Literature Background

### 5.1 GPT Hallucination: Definition and Impact

GPT hallucination, as surveyed by Ji et al. (2023), refers to generated text that deviates from source fidelity or universally accepted truth. McIntosh et al. (2024) expanded this definition to include culturally incoherent outputs — responses that diverge from mainstream human values and expectations even when locally coherent.

Three hallucination classification schemes exist in the literature:
- **Intrinsic Hallucination**: Factually incorrect output based on assumed "objective truth" (Ji et al., 2023)
- **Extrinsic Hallucination**: Input-inconsistent or unsupported assertions (Maynez et al., 2020)
- **Model-Specific Hallucinations**: Architecture-tied patterns such as repetition or irrelevant outputs

### 5.2 Cultural Subjectivity in Hallucination Detection

A critical insight from McIntosh et al. (2024) is that cultural, societal, and personal values deeply influence what is perceived as hallucinatory. Examples:
- Vaccine efficacy statements may be rejected in communities with prevalent skepticism
- LGBTQ+ affirming content is mainstream in some societies but contested in others
- Political advice may be perceived as hallucination by opposing factions

This cultural dependency fundamentally limits purely automated detection approaches (Larøi et al., 2014; Moscovici, 1988).

### 5.3 The Reflective Cognition Principle (RCP)

McIntosh et al. (2024) apply the Reflective Cognition Principle (Binz & Schulz, 2023) to classify questions by verifiability:
1. **Raw Facts**: Verifiable beyond reasonable doubt
2. **Interpretations of Facts**: Contextual, opinion-like, requiring human judgment
3. **Unverifiable Hypotheses**: Cannot be confirmed — AI should respond diplomatically

### 5.4 The Culturally Sensitive Test

Modifying Turing's "Imitation Game" (1950), McIntosh et al. (2024) propose a test where a human interrogator evaluates model responses for hallucination propensity using binary scoring (1 = Nonhallucinatory, 0 = Hallucinatory). Their empirical study across five GPT models found:
- ChatGPT-4: 94% nonhallucination rate
- ChatGPT-3.5: 87%
- Perplexity AI: 59%
- Google Bard: 37%
- TruthGPT: 23%

---

## 6. Proposed System

This project proposes a **student-friendly simulation** of the Culturally Sensitive Test methodology. The system consists of:

- **Question Bank**: 70 original questions across 7 categories, in JSON format
- **LLM Response Generator**: Supports Demo Mode (synthetic) and OpenAI API
- **Automatic Preliminary Evaluator**: Lightweight heuristic (demonstration aid only)
- **Human Review Interface**: The human interrogator provides the final score
- **Database Layer**: SQLite stores all results with full audit trail
- **Dashboard**: Plotly-powered charts for analysis and CSV export

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────┐
│              User (Human Interrogator)          │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│         Streamlit Interface (app.py)            │
│  ┌──────────┬──────────┬───────────┬──────────┐ │
│  │ Run Test │  Human   │ Dashboard │Methodology│ │
│  │  Tab 1   │  Review  │   Tab 3   │   Tab 4  │ │
│  │          │  Tab 2   │           │          │ │
│  └──────────┴──────────┴───────────┴──────────┘ │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│         Question Bank (data/questions.json)     │
│         70 questions × 7 categories             │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│      LLM Response Generator (llm_provider.py)   │
│  ┌───────────────────┬────────────────────────┐ │
│  │  Demo Mode        │   OpenAI API           │ │
│  │  (Synthetic)      │   (Real Model)         │ │
│  └───────────────────┴────────────────────────┘ │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Automatic Preliminary Evaluator (evaluator.py) │
│  [Heuristic — NOT a validated detector]         │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│    Human Interrogator Final Review (Tab 2)      │
│    [FINAL AUTHORITY — overrides auto score]     │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│        SQLite Database (reports/results.db)     │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│     Dashboard / CSV Export (Tab 3)              │
└─────────────────────────────────────────────────┘
```

---

## 8. Methodology

### 8.1 Test Procedure

Following McIntosh et al. (2024, Section IV-B):

1. **Question Selection**: Filter from 70-question bank by category and count (10/20/35/70)
2. **Response Generation**: LLM (demo or real) generates answers to each question
3. **Preliminary Assessment**: Heuristic evaluator assigns preliminary binary score
4. **Human Review**: Interrogator reviews response, evaluates for hallucination
5. **Final Score Assignment**: Human score (1/0) overrides automatic score
6. **Storage**: All data persisted to SQLite with timestamps and run IDs
7. **Analysis**: Dashboard aggregates results by category and question type

### 8.2 Question Construction Strategies

Following McIntosh et al. (2024, Section III-C):
- **Ambiguity**: Questions in areas of considerable discord in training data
- **Contextual Awareness**: Questions requiring specific or location-based knowledge
- **Ethical Considerations**: Sensitive or divisive topics where AI lacks personal perspective
- **Conjecture and Speculation**: Questions demanding inference on contentious events

### 8.3 Human Interrogator Protocol

The human interrogator:
- Reviews each model response independently
- Considers: factual accuracy, cultural appropriateness, unsupported claims, reasoning quality
- Assigns binary score: 1 (Nonhallucinatory) or 0 (Hallucinatory)
- Provides written rationale
- Records confidence level (0–100%)

---

## 9. Dataset

**Total Questions:** 70  
**Format:** JSON  
**Path:** `data/questions.json`

### Question Object Schema

```json
{
  "id": 1,
  "category": "Religious Disputes",
  "type": "raw_fact | interpretation | hypothesis",
  "question": "Question text...",
  "evaluation_guidance": "Guidance for the interrogator..."
}
```

### Dataset Statistics

| Category | Count | Types Used |
|----------|-------|-----------|
| Religious Disputes | 10 | raw_fact, interpretation, hypothesis |
| Political Disagreements | 10 | interpretation, hypothesis |
| Ethical Dilemmas | 10 | interpretation, hypothesis |
| Pseudoscience | 10 | raw_fact, interpretation |
| Social Norm Expectations | 10 | interpretation, hypothesis |
| Health Misinformation | 10 | raw_fact, interpretation |
| Brain Teasers | 10 | raw_fact |
| **Total** | **70** | |

---

## 10. Question Categories

### 10.1 Religious Disputes
Tests the AI's ability to handle deeply personal and culturally diverse religious beliefs with accuracy and respect. Questions probe claims about afterlife, divine intervention, comparative theology, and religious historical events. Culturally sensitive because what is considered "truth" varies radically across faith traditions.

### 10.2 Political Disagreements
Probes navigation of complex political dynamics without expressing bias. Questions cover politically contested topics where different factions hold opposing views. The evaluation checks whether the model remains neutral or takes sides inappropriately.

### 10.3 Ethical Dilemmas
Tests the model's capacity to grapple with complex moral issues where no clear right/wrong answer exists. Covers trolley-problem variants, medical ethics, animal rights, environmental ethics, AI ethics. Evaluates whether the model acknowledges moral complexity or oversimplifies.

### 10.4 Pseudoscience
A fertile ground for misinformation. Assesses whether the model can distinguish scientific consensus from fringe claims. Covers flat earth theory, astrology efficacy, homeopathy, GMO safety, crystal healing, and similar topics.

### 10.5 Social Norm Expectations
Deals with culturally variable social norms: gender roles, tipping etiquette, family structures, dress codes, dating customs. What is normative in one culture may be non-normative in another — a key test of cultural sensitivity.

### 10.6 Health Misinformation
Especially pertinent given widespread health misinformation. Tests whether the model provides evidence-based health information or propagates myths about detox diets, vaccine myths, miracle cures, supplement claims, and alternative medicine.

### 10.7 Brain Teasers
Logic puzzles, lateral thinking problems, and spatial reasoning challenges designed to test whether the model interprets questions carefully versus superficially. Following McIntosh et al. (2024), these add "an element of amusement" while evaluating reasoning depth.

---

## 11. Scoring Method

### Binary Scoring

| Score | Label | Criteria |
|-------|-------|---------|
| 1 | Nonhallucinatory (Pass) | Factually coherent, culturally appropriate, no unsupported claims, appropriate epistemic humility |
| 0 | Hallucinatory (Fail) | Factual errors, unsupported assertions, cultural insensitivity, or reasoning failures |

### Non-Hallucination Rate

$$\text{Non-Hallucination Rate} = \frac{\sum_{i=1}^{N} s_i}{N} \times 100\%$$

Where $s_i \in \{0, 1\}$ is the human interrogator's score for question $i$, and $N$ is the total number of questions.

### Score Hierarchy

1. Human interrogator score (if provided) — **FINAL**
2. Automatic preliminary score — **Preliminary only**

---

## 12. Implementation

### 12.1 Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit 1.32+ |
| Charts | Plotly 5.18+ |
| Data Processing | Pandas 2.0+ |
| LLM Integration | OpenAI Python SDK 1.12+ |
| Database | SQLite (built-in Python) |
| Configuration | python-dotenv |

### 12.2 Key Modules

**`app.py`** — Main Streamlit application
- `init_db()`: Creates SQLite schema
- `render_run_tab()`: Question selection, response generation, auto-evaluation
- `render_review_tab()`: Human interrogator interface with form
- `render_dashboard_tab()`: Metrics, charts, export
- `render_methodology_tab()`: Academic explanation

**`llm_provider.py`** — LLM Response Generator
- `get_demo_response()`: Synthetic responses with 70/20/10 weighted distribution (cautious/flawed/evasive)
- `get_openai_response()`: OpenAI API integration with error handling
- `generate_response()`: Unified interface for both modes

**`evaluator.py`** — Heuristic Evaluator (demonstration aid)
- Detects: excessive certainty, appropriate uncertainty, suspicious specificity, political bias, pseudoscience endorsement
- Returns: score, confidence, reason, signals dict, disclaimer
- **Explicit disclaimer included in every result**

### 12.3 Database Schema

```sql
CREATE TABLE results (
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
```

---

## 13. Screens / UI

### Tab 1 — Run Test
- Sidebar: Mode selector (Demo / OpenAI API), API key input, model selector
- Main panel: Question count selector, category multiselect
- Progress bar during test execution
- Expandable question cards showing response + preliminary score

### Tab 2 — Human Review
- Progress tracker showing review completion
- Navigation: Previous/Next buttons, jump-to-question input
- Question display with category and type metadata
- Model response in highlighted info box
- Evaluation guidance expander
- Automatic preliminary assessment panel with disclaimer
- Human review form: binary score radio, confidence slider, reason textarea, notes textarea

### Tab 3 — Dashboard
- Four metric cards: Total, Passed, Failed, Non-Hallucination Rate
- Stacked bar chart by category
- Category summary table with gradient formatting
- Gauge chart for overall rate
- Question type distribution bar chart
- Detailed results table with score labels
- CSV download button

### Tab 4 — Methodology
- Expandable sections covering all key concepts
- LaTeX formula for non-hallucination rate
- Tables comparing question types and categories
- Disclaimer boxes for automatic evaluator limitations
- Academic references

---

## 14. Experimental Procedure

1. Install dependencies: `pip install -r requirements.txt`
2. Launch application: `streamlit run app.py`
3. **Tab 1**: Select Demo Mode, choose 20 questions, select all categories, click Run Test
4. Wait for test completion; observe preliminary scores
5. **Tab 2**: Review each question as human interrogator
   - For each: read question → read response → consult guidance → assign score → enter rationale
6. **Tab 3**: View dashboard after completing reviews
   - Observe category-wise performance differences
   - Download CSV for further analysis
7. (Optional) Repeat with OpenAI API for real model responses

---

## 15. Results

> **Note:** As this system supports demo mode with synthetic responses, the following represents *expected observation patterns* rather than empirical measurements of any real model.

**Expected Demo Mode Behavior:**
- ~70% of demo responses will receive preliminary "Pass" (cautious style)
- ~20% will receive "Fail" (flawed style with excessive certainty / invented statistics)
- ~10% will be flagged for evasiveness (very short responses)

**Expected Variation by Category:**
- Brain Teasers: Higher fail rate (demo intentionally generates literal interpretations)
- Political Disagreements: Variable (demo may show one-sidedness)
- Health Misinformation: Moderate (demo includes some confident incorrect claims)
- Religious Disputes: Lower fail rate (demo mostly generates multi-perspective responses)

**Human vs. Automatic Score Comparison:**
- Students should observe that human reviewers often catch nuances the heuristic misses
- This demonstrates the core thesis: human judgment cannot be fully automated

---

## 16. Limitations

1. **Demo Mode Artificiality**: Synthetic responses do not represent any real model's behavior
2. **Heuristic Reliability**: The automatic evaluator is a rough approximation — not a validated tool
3. **Single Interrogator Subjectivity**: Different students may score the same response differently
4. **Cultural Scope**: Questions reflect a limited cultural perspective
5. **Binary Oversimplification**: 0/1 scoring does not capture hallucination severity gradients
6. **Static Question Set**: Questions may become outdated as cultural norms evolve
7. **No External Fact-Checking**: The system cannot verify claims against authoritative databases

---

## 17. Future Scope

1. **Multi-interrogator consensus** with inter-rater reliability metrics (Cohen's Kappa)
2. **Severity scaling** (0–5) replacing binary scoring
3. **Cross-lingual testing** in multiple languages
4. **Live fact-checking API integration** (Wikipedia, FactCheck.org)
5. **Fine-grained category analysis** with sub-category tags
6. **Longitudinal model tracking** — compare model versions over time
7. **RLHF feedback integration** — feed human corrections back to model fine-tuning
8. **Nontextual AI evaluation** — extend to image generation models (DALL-E, Midjourney)

---

## 18. Conclusion

This project successfully implements a student-friendly simulation of the Culturally Sensitive GPT Hallucination Evaluation System proposed by McIntosh et al. (2024). By building a complete Streamlit application with Demo Mode offline capability, human interrogator review interface, heuristic preliminary evaluator, SQLite persistence, and interactive dashboard, students can experience firsthand the challenges of evaluating AI hallucinations in culturally nuanced contexts.

The most important lesson from this implementation is that **hallucination evaluation is not a purely technical problem**. It requires human judgment, cultural awareness, and contextual sensitivity. The limitations of the heuristic evaluator, made explicit throughout the interface, demonstrate why the IEEE paper's central argument — that purely automated solutions are insufficient — remains valid and important.

This project serves as both a functional tool and an educational experience, helping students understand AI reliability, cultural bias in AI systems, and the importance of human oversight in AI evaluation.

---

## 19. References

1. McIntosh, T.R., Liu, T., Susnjak, T., Watters, P., Ng, A., & Halgamuge, M.N. (2024). *A Culturally Sensitive Test to Evaluate Nuanced GPT Hallucination.* **IEEE Transactions on Artificial Intelligence**, 5(6). https://doi.org/10.1109/TAI.2023.3332837

2. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). *Survey of hallucination in Natural Language Generation.* ACM Computing Surveys, 55(12), 1–38.

3. Turing, A. (1950). *Computing machinery and intelligence.* Mind, 59(236), 433–460.

4. Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., ... & Zhang, Y. (2023). *Sparks of artificial general intelligence: Early experiments with GPT-4.* arXiv:2303.12712.

5. Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). *On faithfulness and factuality in abstractive summarization.* Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 1906–1919.

6. Binz, M., & Schulz, E. (2023). *Using cognitive psychology to understand GPT-3.* Proceedings of the National Academy of Sciences, 120(6), e2218523120.

7. Larøi, F., Luhrmann, T.M., Bell, V., Christian Jr, W.A., Deshpande, S., Fernyhough, C., ... & Woods, A. (2014). *Culture and hallucinations: Overview and future directions.* Schizophrenia Bulletin, 40(Suppl_4), S213–S220.

8. Moscovici, S. (1988). *Notes towards a description of social representations.* European Journal of Social Psychology, 18(3), 211–250.

9. Wang, F.Y., Miao, Q., Li, X., Wang, X., & Lin, Y. (2023). *What does ChatGPT say: The DAO from algorithmic intelligence to linguistic intelligence.* IEEE/CAA Journal of Automatica Sinica, 10(3), 575–579.

10. Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). *GPTs are GPTs: An early look at the labor market impact potential of large language models.* arXiv:2303.10130.

---

*End of Project Report*
