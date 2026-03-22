"""
classifier.py — AI Opportunity Classifier
==========================================
Takes WorkflowTask objects from parser.py and classifies each task as:

    AUTOMATABLE   — AI can fully replace the human (e.g. "send email", "log data")
    AUGMENTABLE   — AI assists but human stays in the loop (e.g. "review invoice")
    NON_AI        — Human judgment required, AI adds no value (e.g. "negotiate contract")

Strategy : Few-shot classification via facebook/bart-large-mnli
           (HuggingFace Inference API — free tier, no GPU needed)

How few-shot works here
-----------------------
Instead of fine-tuning a model (expensive, needs labelled data), we use
"hypothesis templates" — we ask the model:
  "Does this text entail: [task] is a task that can be fully automated by AI?"
The model returns a probability. We do this for all 3 labels and pick the highest.

The few-shot "examples" are baked into the hypothesis templates as semantic
anchors — the richer the template, the more the model aligns with your domain.

Author  : AI Workflow Optimizer Project
Cost    : $0 — HuggingFace free Inference API (no key needed for public models)
Install : pip install requests pandas
"""

from __future__ import annotations

import os
import time
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum

import requests
import pandas as pd
from dotenv import load_dotenv

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Load .env file automatically (pip install python-dotenv)
load_dotenv()

# Import WorkflowTask from parser (same src/ directory)
try:
    from parser import WorkflowTask
except ImportError:
    from src.parser import WorkflowTask

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HuggingFace Inference API -- correct v1 endpoint (2025)
# /v1 route handles zero-shot classification on free tier (CPU inference)
# Model is specified in the URL path after /models/
HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "facebook/bart-large-mnli/v1/chat/completions"
)
# Zero-shot classification endpoint (correct task endpoint)
HF_ZSC_URL = (
    "https://router.huggingface.co/hf-inference/v1"
)
HF_MODEL = "facebook/bart-large-mnli"

# Your HuggingFace token — get one free at huggingface.co/settings/tokens
# Set as environment variable:  export HF_TOKEN="hf_xxxxxxxxxxxx"
# Or paste directly (not recommended for shared code):
HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # loaded from .env via python-dotenv

# Rate limiting — free tier allows ~30,000 requests/month (~1 req/sec safe)
REQUEST_DELAY_SECONDS: float = 1.2

# Retry config for transient API errors (model loading, rate limits)
MAX_RETRIES: int   = 3
RETRY_WAIT : float = 10.0    # seconds to wait when model is loading

# SQLite database path (project root)
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "workflow_tasks.db"


# ---------------------------------------------------------------------------
# Label enum
# ---------------------------------------------------------------------------
class AILabel(str, Enum):
    AUTOMATABLE = "automatable"
    AUGMENTABLE = "augmentable"
    NON_AI      = "non_ai"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    """
    Output of the classifier for a single WorkflowTask.

    Attributes
    ----------
    task_id         : matches WorkflowTask.task_id
    task_name       : human-readable name
    label           : predicted AILabel
    confidence      : score of the winning label (0.0 – 1.0)
    scores          : full dict of all three label scores
    reasoning       : short human-readable explanation
    tool_suggestion : suggested open-source AI tool for the task
    """
    task_id        : str
    task_name      : str
    label          : AILabel
    confidence     : float
    scores         : dict[str, float]  = field(default_factory=dict)
    reasoning      : str               = ""
    tool_suggestion: str               = ""

    def __repr__(self) -> str:
        return (
            f"ClassificationResult("
            f"id={self.task_id!r}, "
            f"name={self.task_name!r}, "
            f"label={self.label.value!r}, "
            f"confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Few-shot hypothesis templates
# ---------------------------------------------------------------------------
# These are the "examples" in few-shot — rich semantic descriptions that
# anchor the model to your domain (business process automation).
# Each label gets a detailed hypothesis; BART measures entailment probability.

_HYPOTHESIS_TEMPLATES: dict[AILabel, str] = {
    AILabel.AUTOMATABLE: (
        "This task is fully automatable by an AI system without any human involvement. "
        "It involves purely mechanical, rule-based, or data processing actions "
        "such as sending emails, logging records, generating reports from templates, "
        "routing items by rules, extracting data, scheduling jobs, validating fields, "
        "archiving files, calculating values, or matching records."
    ),
    AILabel.AUGMENTABLE: (
        "This task requires a human to make a final decision, but AI can assist "
        "by providing analysis, drafts, or recommendations for the human to review. "
        "Examples: approving requests, reviewing flagged items, assessing risk, "
        "drafting responses, classifying severity, prioritising work, verifying reports, "
        "recommending options, screening candidates, and analysing trends."
    ),
    AILabel.NON_AI: (
        "This task is inherently human and cannot be automated or assisted by AI. "
        "It requires face-to-face trust, emotional intelligence, ethical accountability, "
        "or political judgment. Examples: negotiating contracts, conducting interviews, "
        "mediating conflicts, presenting to a board, building client relationships, "
        "making hiring decisions, leading crisis response, designing ethics policies."
    ),
}

# ---------------------------------------------------------------------------
# Keyword-based pre-classifier
# Handles clear-cut cases with near-perfect precision before calling BART.
# BART is only invoked for tasks that don't match any keyword rule.
# ---------------------------------------------------------------------------

# Verbs that almost always indicate full automation (no human needed)
_AUTO_VERBS = {
    "send", "log", "record", "archive", "schedule", "notify",
    "generate", "calculate", "compute", "extract", "route",
    "validate", "match", "sync", "upload", "download", "backup",
    "export", "import", "trigger", "execute", "run", "process",
    "format", "convert", "transform", "index", "store", "save",
}

# Verbs that almost always indicate human-in-the-loop needed
_AUG_VERBS = {
    "review", "approve", "assess", "evaluate", "verify", "inspect",
    "analyse", "analyze", "audit", "recommend", "suggest", "draft",
    "classify", "prioritise", "prioritize", "investigate", "screen",
    "shortlist", "flag", "escalate", "resolve", "handle",
}

# Verbs/phrases that almost always indicate non-AI
_NON_AI_VERBS = {
    "negotiate", "mediate", "interview", "counsel", "mentor",
    "present", "pitch", "lead", "chair", "facilitate",
    "decide", "strategise", "strategize",
}
_NON_AI_PHRASES = {
    "build trust", "relationship", "hiring decision", "final decision",
    "acquisition strategy", "ethics policy", "crisis communication",
    "performance review", "board presentation", "executive",
}


def _keyword_classify(task_name: str) -> tuple[str, float] | None:
    """
    Attempt to classify a task by keyword matching.
    Returns (label, confidence) if a match is found, else None.

    Confidence is fixed at 0.90 for keyword matches — high but not perfect,
    leaving room for BART to override on genuinely ambiguous tasks.
    """
    name_lower = task_name.lower()
    words      = set(name_lower.split())
    first_verb = name_lower.split()[0] if name_lower.split() else ""

    # Non-AI phrases take highest priority (most distinctive)
    for phrase in _NON_AI_PHRASES:
        if phrase in name_lower:
            return (AILabel.NON_AI.value, 0.90)

    if first_verb in _NON_AI_VERBS or words & _NON_AI_VERBS:
        return (AILabel.NON_AI.value, 0.90)

    # Augmentable verbs next (more specific than automatable)
    if first_verb in _AUG_VERBS or words & _AUG_VERBS:
        return (AILabel.AUGMENTABLE.value, 0.88)

    # Automatable verbs last
    if first_verb in _AUTO_VERBS or words & _AUTO_VERBS:
        return (AILabel.AUTOMATABLE.value, 0.92)

    return None   # No keyword match — fall through to BART



_TOOL_MAP: dict[str, str] = {
    "email"        : "n8n (open-source workflow automation)",
    "send"         : "n8n (open-source workflow automation)",
    "log"          : "Apache Airflow (workflow scheduling)",
    "record"       : "Apache Airflow (workflow scheduling)",
    "generate"     : "Hugging Face text generation pipeline",
    "extract"      : "spaCy NLP extraction pipeline",
    "route"        : "n8n routing workflow",
    "validate"     : "Great Expectations (data validation)",
    "notify"       : "n8n notification automation",
    "schedule"     : "Apache Airflow DAG",
    "upload"       : "n8n file automation",
    "download"     : "n8n file automation",
    "create"       : "LangChain document generation",
    "calculate"    : "Pandas + NumPy automated computation",
    "match"        : "Fuzzy matching (rapidfuzz library)",
    "review"       : "LangChain RAG summarisation assistant",
    "approve"      : "LangChain decision-support assistant",
    "analyse"      : "LangChain RAG analytics assistant",
    "analyze"      : "LangChain RAG analytics assistant",
    "check"        : "LangChain document QA assistant",
    "verify"       : "LangChain document QA assistant",
    "draft"        : "LangChain text generation assistant",
    "resolve"      : "LangChain RAG resolution assistant",
    "handle"       : "LangChain conversational assistant",
    "assess"       : "LangChain RAG assessment assistant",
    "investigate"  : "LangChain RAG research assistant",
    "recommend"    : "LangChain recommendation engine",
    "classify"     : "HuggingFace zero-shot classifier",
    "prioritise"   : "LangChain prioritisation assistant",
    "prioritize"   : "LangChain prioritisation assistant",
    "archive"      : "Apache Airflow (workflow scheduling)",
    "screen"       : "LangChain RAG assessment assistant",
}

_DEFAULT_TOOLS: dict[AILabel, str] = {
    AILabel.AUTOMATABLE: "n8n or Apache Airflow (open-source automation)",
    AILabel.AUGMENTABLE: "LangChain + HuggingFace (open-source AI assistant)",
    AILabel.NON_AI     : "No AI tooling recommended",
}


# ---------------------------------------------------------------------------
# HuggingFace API client
# ---------------------------------------------------------------------------
class HuggingFaceClient:
    """
    Thin wrapper around the HuggingFace free Inference API.

    Handles:
    - Authentication (optional for public models)
    - Retries when the model is still loading (HTTP 503)
    - Rate limiting to stay within the free tier
    - Clear error messages for common failure modes
    """

    def __init__(self, token: str = HF_TOKEN, url: str = HF_ZSC_URL) -> None:
        self._url     = url
        self._headers = {
            "Content-Type": "application/json",
        }
        if not token:
            raise EnvironmentError(
                "HF_TOKEN is required.\n\n"
                "Fix (30 seconds):\n"
                "  1. Go to:  https://huggingface.co/settings/tokens\n"
                "  2. Click 'New token' > role: Read > copy it\n"
                "  3. Add to your .env file:\n"
                "       HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx\n"
                "  (Free accounts get monthly inference credits -- enough for this project)"
            )
        self._headers["Authorization"] = f"Bearer {token}"
        logger.info("HuggingFace token loaded -- authenticated requests enabled.")

    def zero_shot_classify(
        self,
        text      : str,
        hypotheses: list[str],
    ) -> list[float]:
        """
        Run NLI-based few-shot classification via HuggingFace /v1 API.

        Sends all hypotheses in ONE API call using candidate_labels.
        Returns entailment scores in the same order as the input hypotheses.

        Parameters
        ----------
        text       : the task name / description to classify
        hypotheses : list of candidate label descriptions (few-shot anchors)

        Returns
        -------
        list[float] : entailment scores in same order as hypotheses
        """
        # /v1 endpoint: send all candidate labels in a single call
        payload = {
            "model"     : HF_MODEL,
            "inputs"    : text,
            "parameters": {
                "candidate_labels": hypotheses,
                "multi_label"     : True,   # independent scores per label
            },
        }
        response = self._post_with_retry(payload)

        # Response: {"labels": [...], "scores": [...]} in descending score order
        # Re-map back to original hypothesis order
        label_to_score = dict(zip(response["labels"], response["scores"]))
        scores = [label_to_score.get(h, 0.0) for h in hypotheses]

        time.sleep(REQUEST_DELAY_SECONDS)
        return scores

    def _post_with_retry(self, payload: dict) -> dict:
        """POST to HF API with retry logic for model-loading delays."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=30,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    # /v1 may wrap result in a list — unwrap if needed
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                    return data

                elif resp.status_code == 503:
                    # Model is loading — wait and retry
                    wait = resp.json().get("estimated_time", RETRY_WAIT)
                    logger.warning(
                        "Model loading (attempt %d/%d) — waiting %.0fs ...",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(float(wait))

                elif resp.status_code == 429:
                    logger.warning(
                        "Rate limit hit — waiting %ds before retry ...",
                        RETRY_WAIT,
                    )
                    time.sleep(RETRY_WAIT)

                elif resp.status_code == 401:
                    raise PermissionError(
                        "Invalid HuggingFace token. "
                        "Check your HF_TOKEN environment variable."
                    )

                else:
                    resp.raise_for_status()

            except requests.exceptions.Timeout:
                logger.warning(
                    "Request timed out (attempt %d/%d).", attempt, MAX_RETRIES
                )
                time.sleep(RETRY_WAIT)

        raise RuntimeError(
            f"HuggingFace API failed after {MAX_RETRIES} attempts. "
            f"Check your internet connection or try again later."
        )


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------
class WorkflowClassifier:
    """
    Classifies WorkflowTask objects using few-shot NLI via BART.

    Usage
    -----
        # use_local=True  -> runs entirely on your machine (recommended)
    # use_local=False -> uses HuggingFace API (needs valid fine-grained token)
    classifier = WorkflowClassifier(use_local=True)
        results = classifier.classify_all(tasks)
        classifier.save_to_db(results)
        df = classifier.to_dataframe(results)
    """

    def __init__(self, hf_token: str = HF_TOKEN, use_local: bool = True) -> None:
        """
        Parameters
        ----------
        use_local : if True (default), run bart-large-mnli locally via
                    transformers pipeline -- no API, no token, no rate limits.
                    Falls back to HuggingFace API if transformers not installed.
        hf_token  : only used when use_local=False
        """
        self._labels     = list(AILabel)
        self._hypotheses = [_HYPOTHESIS_TEMPLATES[label] for label in self._labels]
        self._label_names = [label.value for label in self._labels]

        # Short readable label names for local pipeline candidate_labels
        self._short_labels = {
            AILabel.AUTOMATABLE: "automatable by AI",
            AILabel.AUGMENTABLE: "requires human oversight with AI help",
            AILabel.NON_AI     : "requires human judgment only",
        }

        if use_local and TRANSFORMERS_AVAILABLE:
            logger.info(
                "Loading bart-large-mnli locally via transformers pipeline "
                "(first run downloads ~1.6GB -- cached after that) ..."
            )
            self._local_pipe = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,          # CPU -- no GPU needed
            )
            self._use_local = True
            logger.info("Local pipeline ready -- no API calls needed.")
        else:
            if use_local and not TRANSFORMERS_AVAILABLE:
                logger.warning(
                    "transformers not installed -- falling back to HF API. "
                    "Run:  pip install transformers torch"
                )
            self._local_pipe = None
            self._use_local  = False
            self._client     = HuggingFaceClient(token=hf_token)
            logger.info("Using HuggingFace Inference API.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, task: WorkflowTask) -> ClassificationResult:
        """
        Classify a single WorkflowTask.

        Parameters
        ----------
        task : a WorkflowTask from parser.py

        Returns
        -------
        ClassificationResult
        """
        logger.info("Classifying: [%s] %s", task.task_id, task.name)

        # ── Step 1: keyword pre-classifier (fast, high precision) ────────
        keyword_result = _keyword_classify(task.name)
        if keyword_result is not None:
            best_label_str, confidence = keyword_result
            best_label = AILabel(best_label_str)
            score_map  = {
                l.value: round(confidence if l.value == best_label_str else
                               (1 - confidence) / 2, 4)
                for l in self._labels
            }
            logger.debug(
                "  keyword match -> %s (%.2f)", best_label_str, confidence
            )
        else:
            # ── Step 2: BART NLI fallback for ambiguous tasks ─────────────
            logger.debug("  no keyword match -> calling BART")
            input_text = self._build_input_text(task)

            if self._use_local:
                result = self._local_pipe(
                    input_text,
                    candidate_labels=self._hypotheses,
                    multi_label=True,
                )
                hyp_to_label = {
                    hyp: label.value
                    for label, hyp in zip(self._labels, self._hypotheses)
                }
                score_map = {
                    hyp_to_label[lbl]: round(score, 4)
                    for lbl, score in zip(result["labels"], result["scores"])
                }
            else:
                raw_scores = self._client.zero_shot_classify(
                    text      = input_text,
                    hypotheses= self._hypotheses,
                )
                score_map = {
                    label.value: round(score, 4)
                    for label, score in zip(self._labels, raw_scores)
                }

            # Pick winner — if gap < 0.20, default to augmentable (safe middle)
            sorted_labels    = sorted(score_map, key=score_map.__getitem__, reverse=True)
            best_label_str   = sorted_labels[0]
            second_label_str = sorted_labels[1]
            gap = score_map[best_label_str] - score_map[second_label_str]

            if gap < 0.20:
                best_label_str = AILabel.AUGMENTABLE.value   # safe default

            best_label = AILabel(best_label_str)
            confidence = score_map[best_label_str]

        result = ClassificationResult(
            task_id        = task.task_id,
            task_name      = task.name,
            label          = best_label,
            confidence     = confidence,
            scores         = score_map,
            reasoning      = self._generate_reasoning(task, best_label, confidence),
            tool_suggestion= self._suggest_tool(task.name, best_label),
        )

        logger.info(
            "  → %s (confidence: %.2f)  tool: %s",
            result.label.value, result.confidence, result.tool_suggestion
        )
        return result

    def classify_all(
        self,
        tasks           : list[WorkflowTask],
        skip_low_freq   : int = 0,
        save_to_db      : bool = True,
    ) -> list[ClassificationResult]:
        """
        Classify a list of WorkflowTask objects.

        Parameters
        ----------
        tasks         : list of WorkflowTask from parser.py
        skip_low_freq : skip tasks with frequency below this threshold
                        (useful for filtering noise in large event logs)
        save_to_db    : auto-save results to SQLite after classification

        Returns
        -------
        list[ClassificationResult]
        """
        if not tasks:
            logger.warning("No tasks to classify.")
            return []

        # Optional frequency filter
        if skip_low_freq > 0:
            before = len(tasks)
            tasks = [t for t in tasks if t.frequency >= skip_low_freq]
            logger.info(
                "Frequency filter (>= %d): %d → %d tasks",
                skip_low_freq, before, len(tasks)
            )

        results: list[ClassificationResult] = []
        total = len(tasks)

        logger.info("Starting classification of %d tasks ...", total)

        for i, task in enumerate(tasks, start=1):
            logger.info("Progress: %d / %d", i, total)
            try:
                result = self.classify(task)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Failed to classify task '%s': %s — skipping.", task.name, exc
                )

        logger.info(
            "Classification complete: %d/%d tasks labelled.",
            len(results), total
        )

        if save_to_db and results:
            self.save_to_db(results)

        return results

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def to_dataframe(self, results: list[ClassificationResult]) -> pd.DataFrame:
        """Convert results to a clean Pandas DataFrame for analysis."""
        rows = []
        for r in results:
            rows.append({
                "task_id"        : r.task_id,
                "task_name"      : r.task_name,
                "label"          : r.label.value,
                "confidence"     : r.confidence,
                "score_automatable": r.scores.get("automatable", 0.0),
                "score_augmentable": r.scores.get("augmentable", 0.0),
                "score_non_ai"     : r.scores.get("non_ai",      0.0),
                "reasoning"      : r.reasoning,
                "tool_suggestion": r.tool_suggestion,
            })
        df = pd.DataFrame(rows)
        logger.info("DataFrame created: %d rows × %d cols", len(df), len(df.columns))
        return df

    def save_to_db(
        self,
        results: list[ClassificationResult],
        db_path: Path = DB_PATH,
    ) -> None:
        """
        Persist classification results to SQLite.
        Creates the database and table automatically if they don't exist.
        Re-running will update existing rows (upsert by task_id).
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                task_id             TEXT PRIMARY KEY,
                task_name           TEXT,
                label               TEXT,
                confidence          REAL,
                score_automatable   REAL,
                score_augmentable   REAL,
                score_non_ai        REAL,
                reasoning           TEXT,
                tool_suggestion     TEXT,
                classified_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        for r in results:
            cur.execute("""
                INSERT INTO classifications (
                    task_id, task_name, label, confidence,
                    score_automatable, score_augmentable, score_non_ai,
                    reasoning, tool_suggestion
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    label             = excluded.label,
                    confidence        = excluded.confidence,
                    score_automatable = excluded.score_automatable,
                    score_augmentable = excluded.score_augmentable,
                    score_non_ai      = excluded.score_non_ai,
                    reasoning         = excluded.reasoning,
                    tool_suggestion   = excluded.tool_suggestion,
                    classified_at     = CURRENT_TIMESTAMP
            """, (
                r.task_id,
                r.task_name,
                r.label.value,
                r.confidence,
                r.scores.get("automatable", 0.0),
                r.scores.get("augmentable", 0.0),
                r.scores.get("non_ai",      0.0),
                r.reasoning,
                r.tool_suggestion,
            ))

        conn.commit()
        conn.close()
        logger.info(
            "Saved %d classification(s) to SQLite: %s", len(results), db_path
        )

    def load_from_db(self, db_path: Path = DB_PATH) -> pd.DataFrame:
        """Load previously saved classification results from SQLite."""
        if not db_path.exists():
            raise FileNotFoundError(
                f"No database found at {db_path}. "
                f"Run classify_all() first."
            )
        conn = sqlite3.connect(db_path)
        df   = pd.read_sql("SELECT * FROM classifications ORDER BY task_id", conn)
        conn.close()
        logger.info("Loaded %d rows from %s", len(df), db_path)
        return df

    def summary(self, results: list[ClassificationResult]) -> dict:
        """
        Return a summary dict — useful for the ROI engine and dashboard.

        Returns
        -------
        dict with keys:
            total, automatable_count, augmentable_count, non_ai_count,
            automatable_pct, augmentable_pct, non_ai_pct,
            avg_confidence, high_confidence_tasks
        """
        total = len(results)
        if total == 0:
            logger.warning("summary() called with empty results list.")
            return {
                "total": 0,
                "automatable_count": 0, "augmentable_count": 0, "non_ai_count": 0,
                "automatable_pct": 0.0, "augmentable_pct": 0.0, "non_ai_pct": 0.0,
                "avg_confidence": 0.0, "high_confidence_tasks": [],
            }

        label_counts = {label.value: 0 for label in AILabel}
        for r in results:
            label_counts[r.label.value] += 1

        high_conf = [
            r for r in results
            if r.confidence >= 0.75
        ]

        return {
            "total"               : total,
            "automatable_count"   : label_counts["automatable"],
            "augmentable_count"   : label_counts["augmentable"],
            "non_ai_count"        : label_counts["non_ai"],
            "automatable_pct"     : round(label_counts["automatable"] / total * 100, 1),
            "augmentable_pct"     : round(label_counts["augmentable"] / total * 100, 1),
            "non_ai_pct"          : round(label_counts["non_ai"]      / total * 100, 1),
            "avg_confidence"      : round(
                sum(r.confidence for r in results) / total, 3
            ),
            "high_confidence_tasks": [r.task_name for r in high_conf],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_input_text(task: WorkflowTask) -> str:
        """
        Build an enriched input string for the model.
        Adding frequency context helps the model distinguish high-volume
        routine tasks (more likely automatable) from rare exception tasks.
        """
        text = task.name
        if task.frequency > 1:
            text += f" (occurs {task.frequency} times in the process)"
        if task.duration:
            text += f" (average duration: {task.duration:.0f} minutes)"
        return text

    @staticmethod
    def _generate_reasoning(
        task      : WorkflowTask,
        label     : AILabel,
        confidence: float,
    ) -> str:
        """Generate a short human-readable reasoning string."""
        conf_word = (
            "high" if confidence >= 0.75
            else "moderate" if confidence >= 0.50
            else "low"
        )
        explanations = {
            AILabel.AUTOMATABLE: (
                f"'{task.name}' appears to be a repetitive, rule-based task "
                f"with {conf_word} confidence ({confidence:.0%}) that AI can "
                f"fully automate — freeing up human time for higher-value work."
            ),
            AILabel.AUGMENTABLE: (
                f"'{task.name}' involves decision-making or review that benefits "
                f"from AI assistance ({conf_word} confidence: {confidence:.0%}), "
                f"but human oversight remains important for quality and accountability."
            ),
            AILabel.NON_AI: (
                f"'{task.name}' requires uniquely human judgment, trust, or "
                f"contextual expertise ({conf_word} confidence: {confidence:.0%}). "
                f"AI involvement is not recommended for this task."
            ),
        }
        return explanations[label]

    @staticmethod
    def _suggest_tool(task_name: str, label: AILabel) -> str:
        """Map task keywords to an open-source tool suggestion."""
        if label == AILabel.NON_AI:
            return _DEFAULT_TOOLS[AILabel.NON_AI]

        task_lower = task_name.lower()
        for keyword, tool in _TOOL_MAP.items():
            if keyword in task_lower:
                return tool

        return _DEFAULT_TOOLS[label]


# ---------------------------------------------------------------------------
# Smoke test  (run: python src/classifier.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from parser import WorkflowTask

    # Simulate the 42 BPI 2019 activities as WorkflowTask objects
    sample_tasks = [
        WorkflowTask("task_001", "Create purchase order item",         "xes", frequency=251734),
        WorkflowTask("task_002", "Record goods receipt",               "xes", frequency=89301),
        WorkflowTask("task_003", "Vendor creates invoice",             "xes", frequency=78432),
        WorkflowTask("task_004", "Clear invoice",                      "xes", frequency=67210),
        WorkflowTask("task_005", "Send invoice for payment",           "xes", frequency=55001),
        WorkflowTask("task_006", "Approve purchase order",             "xes", frequency=43800),
        WorkflowTask("task_007", "Remove payment block",               "xes", frequency=32100),
        WorkflowTask("task_008", "Negotiate contract with vendor",     "xes", frequency=1200),
        WorkflowTask("task_009", "Resolve compliance exception",       "xes", frequency=980),
        WorkflowTask("task_010", "Validate GR-based invoice match",    "xes", frequency=44500),
    ]

    print("\n" + "=" * 65)
    print("  CLASSIFIER SMOKE TEST  (few-shot, BART)")
    print("=" * 65)
    print("  Note: first call may take 20–30s while HuggingFace loads the model")
    print()

    # use_local=True  -> runs entirely on your machine (recommended)
    # use_local=False -> uses HuggingFace API (needs valid fine-grained token)
    classifier = WorkflowClassifier(use_local=True)

    # Classify all sample tasks (skip_low_freq=500 to keep demo fast)
    results = classifier.classify_all(
        tasks         = sample_tasks,
        skip_low_freq = 0,
        save_to_db    = True,
    )

    # Print results table
    print("\n" + "=" * 65)
    print(f"  {'TASK':<40} {'LABEL':<14} {'CONF':>5}")
    print("=" * 65)
    for r in results:
        print(f"  {r.task_name:<40} {r.label.value:<14} {r.confidence:>5.2f}")

    # Print summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    s = classifier.summary(results)
    if not s.get('total'):
        print("  No tasks were classified. Check your HF_TOKEN and connection.")
    else:
        print(f"  Total tasks classified : {s['total']}")
    print(f"  Automatable            : {s['automatable_count']}  ({s['automatable_pct']}%)")
    print(f"  Augmentable            : {s['augmentable_count']}  ({s['augmentable_pct']}%)")
    print(f"  Non-AI                 : {s['non_ai_count']}  ({s['non_ai_pct']}%)")
    print(f"  Avg confidence         : {s['avg_confidence']}")
    print(f"  High-confidence tasks  : {s['high_confidence_tasks']}")

    # Export DataFrame
    df = classifier.to_dataframe(results)
    print(f"\n  DataFrame shape: {df.shape}")
    if not df.empty:
        print(df[["task_name", "label", "confidence", "tool_suggestion"]].to_string(index=False))
    else:
        print("  (No results to display -- check token and connection)")

    print("\nclassifier.py smoke-test complete.")
    print(f"Results saved to: {DB_PATH}")