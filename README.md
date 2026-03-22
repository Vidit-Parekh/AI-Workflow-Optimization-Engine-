# 🧠 AI Workflow Optimization Engine

> **Harvard Business School Case Study Implementation**
> *Generative AI in Business Workflows — Where should AI be applied to maximize ROI?*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BART--large--mnli-yellow?style=flat-square)](https://huggingface.co/facebook/bart-large-mnli)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Cost](https://img.shields.io/badge/API%20Cost-%240-brightgreen?style=flat-square)]()

---

## 📌 Overview

This project answers the core question from the **HBS case "Generative AI and the Future of Work"**:

> *"Where in a business workflow should GenAI be applied to maximize ROI?"*

Companies are randomly adopting AI tools without seeing consistent returns. This system takes any business workflow — described in plain English or as a structured event log — and produces a consulting-grade analysis that classifies every task by AI opportunity, estimates financial ROI, and simulates adoption scenarios.

**Built entirely with open-source tools. Zero API cost. Runs locally.**

---

## 🏗️ System Architecture

![System Architecture](Documentation_images/ai_workflow_project_architecture.svg)

The engine is a 4-layer pipeline:

| Layer | Module | What it does |
|---|---|---|
| Input | `parser.py` | Accepts free text or XES/CSV event logs |
| Classification | `classifier.py` | Labels each task: automatable / augmentable / non-AI |
| ROI | `roi_engine.py` | Computes cost saved, hours freed, payback period |
| Output | `app.py` | Streamlit dashboard with charts, graphs, exports |

---

## 🎯 Key Features

- **Dual input modes** — plain English workflow description OR structured PM4Py event log (XES, XML, CSV)
- **Hybrid classifier** — keyword pre-classification with BART-large-mnli fallback for ambiguous tasks
- **ROI engine** — per-task cost saving, hours freed, accuracy delta, composite ROI score
- **What-if simulation** — 5 adoption scenarios from 0% to 100% AI coverage
- **Workflow graph** — interactive PyVis DAG coloured by AI opportunity label
- **SQLite persistence** — all results saved locally, no cloud database needed
- **CSV export** — download ROI report and classification details
- **Free deployment** — runs on Streamlit Community Cloud at zero cost

---

## 📊 Datasets

### Primary — BPI Challenge 2019 (Purchase Order Handling)

| Property | Value |
|---|---|
| Source | [4TU Research Data](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1) |
| Cases | 251,734 purchase order items |
| Events | 1,595,923 |
| Activities | 42 unique workflow tasks |
| Users | 627 (607 human, 20 automated batch processes) |
| Format | XES / XML |

### Additional Kaggle Datasets (in `data/`)

| Dataset | Domain | Format |
|---|---|---|
| Incident Management CSV | IT Service Desk | CSV |
| Insurance Claims Event Log | Insurance | CSV |
| AI Workforce & Automation 2015–2025 | Cross-industry | CSV |
| Global AI Workforce Automation | Cross-industry | CSV |

Place all data files in the `data/` directory (gitignored).

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-workflow-optimizer.git
cd ai-workflow-optimizer
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your HuggingFace token (optional)
```

`.env` file:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
```

### 5. Run the dashboard

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`.

---

## 📁 Project Structure

```
ai-workflow-optimizer/
│
├── src/
│   ├── app.py                        # Streamlit dashboard entry point
│   ├── classifier.py                 # Hybrid keyword + BART classifier
│   ├── parser.py                     # Input handler — text + XES/CSV
│   ├── roi_engine.py                 # ROI estimation engine
│   └── test_classifier_accuracy.py   # 30-task accuracy test suite
│
├── data/                             # gitignored — add your data files here
│   ├── BPI_Challenge_2019.xml
│   ├── Incident_Management_CSV.csv
│   ├── Insurance_claims_event_log.csv
│   ├── global_ai_workforce_automation_2015.csv
│   ├── accuracy_test_results.csv     # auto-generated
│   └── workflow_tasks.db             # auto-generated SQLite database
│
├── Documentation_images/             # Architecture and diagnosis diagrams
│   ├── ai_workflow_project_architecture.svg
│   ├── accuracy_diagnosis.svg
│   ├── real_problem_diagnosis.svg
│   └── student_friendly_ai_workflow_engine.svg
│
├── zips/                             # gitignored — raw downloaded datasets
├── lib/                              # gitignored — PyVis JS dependencies
├── venv/                             # gitignored
│
├── .env                              # gitignored — your secrets
├── .env.example                      # safe template to commit
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧩 Module Deep Dive

### `parser.py` — Input Handler

Accepts two input types and unifies them into `WorkflowTask` objects.

```python
from src.parser import WorkflowParser

parser = WorkflowParser()

# From plain English text
tasks = parser.parse_text("First, the team receives an email...")

# From BPI 2019 XES file
tasks = parser.parse_data_dir("BPI_Challenge_2019.xml")

# Auto-discover all files in data/
tasks = parser.parse_data_dir()
```

**Supported formats:** `.xes` · `.xml` (XES format) · `.csv` (PM4Py compatible)

---

### `classifier.py` — Hybrid AI Classifier

![Accuracy Diagnosis](Documentation_images/accuracy_diagnosis.svg)

Two-stage classification pipeline:

```
Task name → Keyword pre-classifier → Label (if match found)
                ↓ (no match)
           BART-large-mnli NLI → Label (fallback)
```

**Stage 1 — Keyword rules** (~90% of tasks):
- `_AUTO_VERBS`: send, log, generate, calculate, extract, schedule, validate, archive...
- `_AUG_VERBS`: review, approve, assess, draft, verify, classify, recommend...
- `_NON_AI_VERBS`: negotiate, mediate, interview, present, decide...

**Stage 2 — BART NLI** (ambiguous tasks only): `facebook/bart-large-mnli` via local `transformers` pipeline. Zero API calls after first download (~1.6 GB cached).

```python
from src.classifier import WorkflowClassifier

classifier = WorkflowClassifier(use_local=True)
results    = classifier.classify_all(tasks)
summary    = classifier.summary(results)
```

---

### `roi_engine.py` — ROI Estimation Engine

| Metric | Formula |
|---|---|
| Time saved / year | `frequency × duration × saving_factor` |
| Cost saved / year | `time_saved × hourly_rate` |
| Accuracy delta | 15pp for automatable, 8pp for augmentable |

**Saving factors:** automatable → 80% · augmentable → 40% · non_ai → 0%

```python
from src.roi_engine import ROIEngine, ROIConfig

config   = ROIConfig(hourly_rate_usd=55.0, implementation_cost_usd=40_000.0)
engine   = ROIEngine(config=config)
roi_list = engine.compute(results, task_frequencies=frequencies)
whatif   = engine.what_if_simulation(roi_list)
```

---

## 📈 Dashboard Walkthrough

**Sidebar** — Upload `.xml`/`.xes`/`.csv` or paste text. Adjust hourly rate, implementation cost, frequency scale. Click Run Analysis.

**KPI Cards** — Annual saving · Hours freed · Payback period · AI-ready %

**AI Opportunity Heatmap** — Bubble scatter: x = confidence, y = label, size = confidence. Instantly shows certainty distribution.

**ROI Bar Chart** — Ranked by annual saving, coloured by label.

**Workflow Graph** — Interactive PyVis DAG. Node size = frequency. Colour = label.

**What-If Simulation** — Dual-axis chart across 5 adoption scenarios (0–100%).

**Data Table + Export** — Full results in two tabs with CSV download.

---

## 🧪 Test Cases

| # | Industry | Hourly rate | Impl. cost | Expected saving |
|---|---|---|---|---|
| 1 | Enterprise Procurement | $55 | $40,000 | ~$123,750 |
| 2 | Hospital Patient Management | $65 | $75,000 | ~$146,250 |
| 3 | E-commerce Fulfilment | $35 | $25,000 | ~$61,250 |
| 4 | HR Recruitment Pipeline | $50 | $30,000 | ~$71,250 |
| 5 | Bank Loan Processing | $70 | $100,000 | ~$183,750 |

Run the accuracy test suite:

```bash
python src/test_classifier_accuracy.py
```

---

## 🔬 BPI 2019 Results (Full Dataset)

| Metric | Value |
|---|---|
| Total tasks classified | 42 |
| Automatable | ~12 (29%) |
| Augmentable | ~24 (57%) |
| Non-AI | ~6 (14%) |
| Total annual cost saving | ~$8.7M |
| Hours freed / year | ~193,000 hrs |
| Payback period | ~0.1 months |
| Avg classifier confidence | 0.87 |

---

## 🛠️ Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| NLP | spaCy `en_core_web_sm` | Task extraction from text |
| Process mining | PM4Py | XES/CSV event log parsing |
| ML model | BART-large-mnli | NLI zero-shot classification |
| ML framework | HuggingFace Transformers | Local model inference |
| Data | Pandas, NumPy | ROI calculations |
| Graph | NetworkX + PyVis | Workflow DAG visualisation |
| Dashboard | Streamlit | Web UI |
| Charts | Plotly | Interactive visualisations |
| Database | SQLite | Local result persistence |
| Env | python-dotenv | Secret management |

**Total API cost: $0**

---

## 🌐 Deployment

```bash
# Push to GitHub
git add .
git commit -m "Initial deployment"
git push origin main

# Go to share.streamlit.io
# Connect repo → set main file: src/app.py
# Add secrets: HF_TOKEN = "hf_xxxx"
# Deploy — get a public URL instantly
```

> First deploy downloads BART (~1.6 GB). Takes 3–5 minutes. Subsequent runs use cache.

---

## 📚 Academic References

- **HBS Case:** *Generative AI and the Future of Work* — Harvard Business School, 2023
- **Dataset:** van Dongen, B. (2019). *BPI Challenge 2019*. 4TU. [DOI: 10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1)
- **Model:** Lewis, M. et al. (2019). *BART: Denoising Sequence-to-Sequence Pre-training*. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)
- **Process Mining:** van der Aalst, W. (2016). *Process Mining: Data Science in Action*. Springer.
- **McKinsey:** *A future that works: Automation, employment, and productivity* (2017). McKinsey Global Institute.

---

## 📄 Resume Bullet

> Built a consulting-grade **AI Workflow Optimization Engine** inspired by HBS GenAI case study — classifies business process tasks as automatable/augmentable/non-AI using a hybrid keyword + BART-large-mnli pipeline, estimates annual ROI via cost-time modelling ($8.7M on BPI 2019 dataset), and visualises results in a Streamlit dashboard with workflow graph, opportunity heatmap, and what-if simulation. Zero API cost — runs entirely on open-source tools.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built as part of the Harvard Business School AI case study series</sub>
</div>