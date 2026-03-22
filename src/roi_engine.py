"""
roi_engine.py — ROI Estimation Engine
======================================
Takes ClassificationResult objects from classifier.py and computes
the financial and operational return on investment of AI adoption.

Three metrics per task
-----------------------
  TIME SAVED      — hours per year freed up by AI automation/augmentation
  COST SAVED      — monetary value of that time (hourly rate × hours saved)
  ACCURACY DELTA  — estimated error reduction (%) from AI assistance

Aggregation
-----------
  Per-task results roll up to:
  - Total annual cost saving
  - Total hours saved per year
  - Payback period (months to recover AI implementation cost)
  - What-if simulation: 0% vs 50% vs 100% AI adoption

Author  : AI Workflow Optimizer Project
Cost    : $0 — pure Python, Pandas, NumPy. No API calls.
Install : pip install pandas numpy
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from classifier import ClassificationResult, AILabel
except ImportError:
    from src.classifier import ClassificationResult, AILabel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default assumption constants
# All values are conservative industry estimates — override via ROIConfig
# ---------------------------------------------------------------------------

# Average fully-loaded hourly cost of a knowledge worker (USD)
# Source: BLS + McKinsey estimates for ops/admin roles
DEFAULT_HOURLY_RATE_USD: float = 45.0

# Working hours per year (50 weeks × 40 hrs)
WORKING_HOURS_PER_YEAR: int = 2000

# Fraction of time saved per label when AI is applied
# Automatable: AI does the whole task → 80% time saving
# Augmentable: AI assists → 40% time saving
# Non-AI     : no saving
TIME_SAVING_FACTORS: dict[str, float] = {
    AILabel.AUTOMATABLE.value: 0.80,
    AILabel.AUGMENTABLE.value: 0.40,
    AILabel.NON_AI.value     : 0.00,
}

# Accuracy improvement delta (percentage points) when AI is applied
# Based on published RPA/AI studies in process automation
ACCURACY_DELTA: dict[str, float] = {
    AILabel.AUTOMATABLE.value: 15.0,   # rule-based AI nearly eliminates errors
    AILabel.AUGMENTABLE.value:  8.0,   # AI assistance reduces errors moderately
    AILabel.NON_AI.value     :  0.0,
}

# One-time AI implementation cost estimate (USD)
# Conservative estimate for a small-to-medium enterprise deployment
DEFAULT_IMPLEMENTATION_COST_USD: float = 50_000.0

# SQLite DB path
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "workflow_tasks.db"


# ---------------------------------------------------------------------------
# Configuration dataclass — override any assumption cleanly
# ---------------------------------------------------------------------------
@dataclass
class ROIConfig:
    """
    All tuneable parameters for the ROI engine.
    Pass a custom ROIConfig to override any default assumption.

    Example
    -------
        config = ROIConfig(
            hourly_rate_usd=60.0,           # senior analyst rate
            working_hours_per_year=1800,    # accounting for leave
            implementation_cost_usd=30000,  # smaller deployment
        )
        engine = ROIEngine(config=config)
    """
    hourly_rate_usd            : float = DEFAULT_HOURLY_RATE_USD
    working_hours_per_year     : int   = WORKING_HOURS_PER_YEAR
    implementation_cost_usd    : float = DEFAULT_IMPLEMENTATION_COST_USD
    time_saving_factors        : dict  = field(
        default_factory=lambda: dict(TIME_SAVING_FACTORS)
    )
    accuracy_deltas            : dict  = field(
        default_factory=lambda: dict(ACCURACY_DELTA)
    )
    # Adoption scenarios for what-if simulation (fraction of tasks AI-enabled)
    adoption_scenarios         : list  = field(
        default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.0]
    )


# ---------------------------------------------------------------------------
# Per-task ROI result
# ---------------------------------------------------------------------------
@dataclass
class TaskROI:
    """
    ROI breakdown for a single workflow task.

    Attributes
    ----------
    task_id              : matches ClassificationResult.task_id
    task_name            : human-readable name
    label                : AILabel value
    confidence           : classifier confidence score
    frequency            : how many times per year this task occurs
    avg_duration_hours   : average time to complete one instance (hours)
    time_saved_hours_yr  : hours saved per year if AI applied
    cost_saved_usd_yr    : dollar value of time saved per year
    accuracy_delta_pct   : estimated error reduction (percentage points)
    tool_suggestion      : recommended open-source tool
    roi_score            : composite score (0–100) for prioritisation
    """
    task_id             : str
    task_name           : str
    label               : str
    confidence          : float
    frequency           : int
    avg_duration_hours  : float
    time_saved_hours_yr : float
    cost_saved_usd_yr   : float
    accuracy_delta_pct  : float
    tool_suggestion     : str   = ""
    roi_score           : float = 0.0

    def __repr__(self) -> str:
        return (
            f"TaskROI(name={self.task_name!r}, label={self.label!r}, "
            f"cost_saved=${self.cost_saved_usd_yr:,.0f}/yr, "
            f"roi_score={self.roi_score:.1f})"
        )


# ---------------------------------------------------------------------------
# Core ROI engine
# ---------------------------------------------------------------------------
class ROIEngine:
    """
    Computes ROI metrics from classifier results.

    Usage
    -----
        engine  = ROIEngine()
        roi     = engine.compute(results, tasks)
        df      = engine.to_dataframe(roi)
        summary = engine.summary(roi)
        whatif  = engine.what_if_simulation(roi)
        engine.save_to_db(roi)
    """

    def __init__(self, config: ROIConfig | None = None) -> None:
        self.config = config or ROIConfig()
        logger.info(
            "ROIEngine initialised — hourly rate: $%.2f, "
            "working hours/yr: %d, impl. cost: $%.0f",
            self.config.hourly_rate_usd,
            self.config.working_hours_per_year,
            self.config.implementation_cost_usd,
        )

    # ------------------------------------------------------------------
    # Main compute method
    # ------------------------------------------------------------------
    def compute(
        self,
        results         : list[ClassificationResult],
        task_frequencies: dict[str, int]   | None = None,
        task_durations  : dict[str, float] | None = None,
    ) -> list[TaskROI]:
        """
        Compute per-task ROI for a list of ClassificationResults.

        Parameters
        ----------
        results          : output from WorkflowClassifier.classify_all()
        task_frequencies : dict of task_id → annual frequency
                           If None, defaults to 1000 occurrences/year
                           (conservative estimate for enterprise workflows)
        task_durations   : dict of task_id → avg duration in hours
                           If None, defaults to label-based estimates

        Returns
        -------
        list[TaskROI] sorted by cost_saved_usd_yr descending
        """
        if not results:
            logger.warning("No classification results to compute ROI for.")
            return []

        roi_list: list[TaskROI] = []

        for r in results:
            freq     = self._get_frequency(r.task_id, task_frequencies)
            duration = self._get_duration(r.task_id, r.label.value, task_durations)

            time_factor   = self.config.time_saving_factors.get(r.label.value, 0.0)
            accuracy_delta= self.config.accuracy_deltas.get(r.label.value, 0.0)

            # Core ROI formulas
            # Time saved per year = freq × duration × saving_factor
            time_saved_yr = freq * duration * time_factor

            # Cost saved per year = time_saved × hourly_rate
            cost_saved_yr = time_saved_yr * self.config.hourly_rate_usd

            # Composite ROI score (0–100)
            # Weights: 50% cost saving, 30% confidence, 20% accuracy
            roi_score = self._compute_roi_score(
                cost_saved_yr  = cost_saved_yr,
                confidence     = r.confidence,
                accuracy_delta = accuracy_delta,
            )

            roi_list.append(TaskROI(
                task_id            = r.task_id,
                task_name          = r.task_name,
                label              = r.label.value,
                confidence         = r.confidence,
                frequency          = freq,
                avg_duration_hours = duration,
                time_saved_hours_yr= round(time_saved_yr, 2),
                cost_saved_usd_yr  = round(cost_saved_yr, 2),
                accuracy_delta_pct = accuracy_delta,
                tool_suggestion    = r.tool_suggestion,
                roi_score          = round(roi_score, 1),
            ))

        # Sort by cost saved descending — highest ROI tasks first
        roi_list.sort(key=lambda x: x.cost_saved_usd_yr, reverse=True)

        total_saving = sum(r.cost_saved_usd_yr for r in roi_list)
        logger.info(
            "ROI computed for %d tasks — total annual saving: $%s",
            len(roi_list), f"{total_saving:,.0f}"
        )
        return roi_list

    # ------------------------------------------------------------------
    # What-if simulation
    # ------------------------------------------------------------------
    def what_if_simulation(self, roi_list: list[TaskROI]) -> pd.DataFrame:
        """
        Simulate ROI across adoption scenarios.

        For each scenario (0%, 25%, 50%, 75%, 100% of automatable/augmentable
        tasks enabled), compute total annual saving and payback period.

        Returns
        -------
        pd.DataFrame with columns:
            adoption_pct, tasks_enabled, annual_saving_usd,
            hours_saved_yr, payback_months
        """
        if not roi_list:
            return pd.DataFrame()

        # Only automatable and augmentable tasks contribute ROI
        eligible = [r for r in roi_list if r.label != AILabel.NON_AI.value]
        total_eligible = len(eligible)

        rows = []
        for scenario in self.config.adoption_scenarios:
            # How many tasks are enabled at this adoption level
            n_enabled = int(round(total_eligible * scenario))
            # Take the top-N by ROI score (best tasks first)
            enabled_tasks = sorted(
                eligible, key=lambda x: x.roi_score, reverse=True
            )[:n_enabled]

            annual_saving  = sum(t.cost_saved_usd_yr  for t in enabled_tasks)
            hours_saved_yr = sum(t.time_saved_hours_yr for t in enabled_tasks)

            # Payback period = implementation cost / monthly saving
            monthly_saving = annual_saving / 12 if annual_saving > 0 else 0
            payback_months = (
                round(self.config.implementation_cost_usd / monthly_saving, 1)
                if monthly_saving > 0 else None
            )

            rows.append({
                "adoption_pct"      : f"{int(scenario * 100)}%",
                "tasks_enabled"     : n_enabled,
                "annual_saving_usd" : round(annual_saving, 2),
                "hours_saved_yr"    : round(hours_saved_yr, 1),
                "payback_months"    : payback_months,
            })

        df = pd.DataFrame(rows)
        logger.info("What-if simulation complete — %d scenarios.", len(df))
        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self, roi_list: list[TaskROI]) -> dict:
        """
        Return a high-level summary dict for dashboard and reporting.

        Keys
        ----
        total_tasks, automatable_count, augmentable_count, non_ai_count,
        total_cost_saved_usd_yr, total_hours_saved_yr,
        avg_accuracy_delta_pct, payback_months_at_100pct,
        top_3_tasks_by_roi, implementation_cost_usd
        """
        if not roi_list:
            return {}

        auto  = [r for r in roi_list if r.label == AILabel.AUTOMATABLE.value]
        aug   = [r for r in roi_list if r.label == AILabel.AUGMENTABLE.value]
        nonai = [r for r in roi_list if r.label == AILabel.NON_AI.value]

        total_cost_saved  = sum(r.cost_saved_usd_yr   for r in roi_list)
        total_hours_saved = sum(r.time_saved_hours_yr  for r in roi_list)
        avg_accuracy      = (
            sum(r.accuracy_delta_pct for r in roi_list) / len(roi_list)
        )

        monthly_saving   = total_cost_saved / 12
        payback_months   = (
            round(self.config.implementation_cost_usd / monthly_saving, 1)
            if monthly_saving > 0 else None
        )

        top_3 = sorted(roi_list, key=lambda x: x.roi_score, reverse=True)[:3]

        return {
            "total_tasks"              : len(roi_list),
            "automatable_count"        : len(auto),
            "augmentable_count"        : len(aug),
            "non_ai_count"             : len(nonai),
            "total_cost_saved_usd_yr"  : round(total_cost_saved, 2),
            "total_hours_saved_yr"     : round(total_hours_saved, 1),
            "avg_accuracy_delta_pct"   : round(avg_accuracy, 1),
            "payback_months_at_100pct" : payback_months,
            "top_3_tasks_by_roi"       : [t.task_name for t in top_3],
            "implementation_cost_usd"  : self.config.implementation_cost_usd,
        }

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def to_dataframe(self, roi_list: list[TaskROI]) -> pd.DataFrame:
        """Convert TaskROI list to a clean Pandas DataFrame."""
        if not roi_list:
            return pd.DataFrame()
        rows = [
            {
                "task_id"            : r.task_id,
                "task_name"          : r.task_name,
                "label"              : r.label,
                "confidence"         : r.confidence,
                "frequency_per_yr"   : r.frequency,
                "avg_duration_hrs"   : r.avg_duration_hours,
                "time_saved_hrs_yr"  : r.time_saved_hours_yr,
                "cost_saved_usd_yr"  : r.cost_saved_usd_yr,
                "accuracy_delta_pct" : r.accuracy_delta_pct,
                "roi_score"          : r.roi_score,
                "tool_suggestion"    : r.tool_suggestion,
            }
            for r in roi_list
        ]
        df = pd.DataFrame(rows)
        logger.info("ROI DataFrame: %d rows × %d cols", len(df), len(df.columns))
        return df

    def save_to_db(
        self,
        roi_list: list[TaskROI],
        db_path : Path = DB_PATH,
    ) -> None:
        """
        Persist ROI results to SQLite.
        Upserts by task_id — safe to re-run after re-classification.
        """
        if not roi_list:
            logger.warning("No ROI results to save.")
            return

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS roi_results (
                task_id              TEXT PRIMARY KEY,
                task_name            TEXT,
                label                TEXT,
                confidence           REAL,
                frequency_per_yr     INTEGER,
                avg_duration_hrs     REAL,
                time_saved_hrs_yr    REAL,
                cost_saved_usd_yr    REAL,
                accuracy_delta_pct   REAL,
                roi_score            REAL,
                tool_suggestion      TEXT,
                computed_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        for r in roi_list:
            cur.execute("""
                INSERT INTO roi_results (
                    task_id, task_name, label, confidence,
                    frequency_per_yr, avg_duration_hrs,
                    time_saved_hrs_yr, cost_saved_usd_yr,
                    accuracy_delta_pct, roi_score, tool_suggestion
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    label              = excluded.label,
                    confidence         = excluded.confidence,
                    frequency_per_yr   = excluded.frequency_per_yr,
                    avg_duration_hrs   = excluded.avg_duration_hrs,
                    time_saved_hrs_yr  = excluded.time_saved_hrs_yr,
                    cost_saved_usd_yr  = excluded.cost_saved_usd_yr,
                    accuracy_delta_pct = excluded.accuracy_delta_pct,
                    roi_score          = excluded.roi_score,
                    tool_suggestion    = excluded.tool_suggestion,
                    computed_at        = CURRENT_TIMESTAMP
            """, (
                r.task_id, r.task_name, r.label, r.confidence,
                r.frequency, r.avg_duration_hours,
                r.time_saved_hours_yr, r.cost_saved_usd_yr,
                r.accuracy_delta_pct, r.roi_score, r.tool_suggestion,
            ))

        conn.commit()
        conn.close()
        logger.info("Saved %d ROI records to: %s", len(roi_list), db_path)

    def load_from_db(self, db_path: Path = DB_PATH) -> pd.DataFrame:
        """Load previously saved ROI results from SQLite."""
        if not db_path.exists():
            raise FileNotFoundError(
                f"No database at {db_path}. Run compute() + save_to_db() first."
            )
        conn = sqlite3.connect(db_path)
        df   = pd.read_sql(
            "SELECT * FROM roi_results ORDER BY cost_saved_usd_yr DESC", conn
        )
        conn.close()
        logger.info("Loaded %d ROI rows from %s", len(df), db_path)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_frequency(
        task_id         : str,
        task_frequencies: dict[str, int] | None,
    ) -> int:
        """
        Return frequency for a task.

        Priority:
          1. Explicit frequency from event log (most accurate)
          2. If frequency == 1 (text input default), use 500/yr
             as a conservative enterprise estimate
          3. Otherwise use 1000/yr as general default
        """
        if task_frequencies and task_id in task_frequencies:
            freq = task_frequencies[task_id]
            # Text input tasks always have freq=1 — override with enterprise default
            if freq <= 1:
                return 500
            return freq
        return 1000   # conservative default for enterprise workflow

    @staticmethod
    def _get_duration(
        task_id       : str,
        label         : str,
        task_durations: dict[str, float] | None,
    ) -> float:
        """
        Return avg duration (hours) for a task.
        Falls back to label-based estimates if not provided.
        """
        if task_durations and task_id in task_durations:
            return task_durations[task_id]

        # Label-based duration defaults (hours per task instance)
        defaults = {
            AILabel.AUTOMATABLE.value: 0.25,   # 15 min — routine tasks
            AILabel.AUGMENTABLE.value: 0.75,   # 45 min — review/approval tasks
            AILabel.NON_AI.value     : 1.50,   # 90 min — complex human tasks
        }
        return defaults.get(label, 0.5)

    @staticmethod
    def _compute_roi_score(
        cost_saved_yr : float,
        confidence    : float,
        accuracy_delta: float,
    ) -> float:
        """
        Composite ROI score 0–100.

        Weights
        -------
        50% — normalised cost saving (log-scaled to handle large ranges)
        30% — classifier confidence
        20% — accuracy improvement

        The log scale on cost prevents one very high-frequency task from
        dominating the score and masking other valuable opportunities.
        """
        # Normalise cost saving: log scale, cap at 100
        cost_component = min(
            np.log1p(cost_saved_yr) / np.log1p(500_000) * 100, 100
        ) if cost_saved_yr > 0 else 0.0

        confidence_component  = confidence  * 100
        accuracy_component    = min(accuracy_delta * 5, 100)

        score = (
            0.50 * cost_component
          + 0.30 * confidence_component
          + 0.20 * accuracy_component
        )
        return min(score, 100.0)


# ---------------------------------------------------------------------------
# Smoke test  (run: python src/roi_engine.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from classifier import ClassificationResult, AILabel

    # Simulate classifier output matching the smoke test results
    mock_results = [
        ClassificationResult("task_001", "Create purchase order item",
            AILabel.AUGMENTABLE, 0.67, tool_suggestion="LangChain document generation"),
        ClassificationResult("task_002", "Record goods receipt",
            AILabel.AUGMENTABLE, 0.80, tool_suggestion="Apache Airflow"),
        ClassificationResult("task_003", "Vendor creates invoice",
            AILabel.AUGMENTABLE, 0.88, tool_suggestion="LangChain document generation"),
        ClassificationResult("task_004", "Clear invoice",
            AILabel.AUGMENTABLE, 0.92, tool_suggestion="LangChain + HuggingFace"),
        ClassificationResult("task_005", "Send invoice for payment",
            AILabel.AUTOMATABLE, 0.76, tool_suggestion="n8n automation"),
        ClassificationResult("task_006", "Approve purchase order",
            AILabel.AUGMENTABLE, 0.93, tool_suggestion="LangChain decision assistant"),
        ClassificationResult("task_007", "Remove payment block",
            AILabel.AUGMENTABLE, 0.89, tool_suggestion="LangChain + HuggingFace"),
        ClassificationResult("task_008", "Negotiate contract with vendor",
            AILabel.NON_AI,      0.87, tool_suggestion="No AI tooling"),
        ClassificationResult("task_009", "Resolve compliance exception",
            AILabel.AUGMENTABLE, 0.97, tool_suggestion="LangChain RAG"),
        ClassificationResult("task_010", "Validate GR-based invoice match",
            AILabel.AUGMENTABLE, 0.98, tool_suggestion="Great Expectations"),
    ]

    # BPI 2019 actual frequencies from the dataset
    frequencies = {
        "task_001": 251734, "task_002": 89301, "task_003": 78432,
        "task_004": 67210,  "task_005": 55001, "task_006": 43800,
        "task_007": 32100,  "task_008": 1200,  "task_009": 980,
        "task_010": 44500,
    }

    # Run ROI engine
    engine   = ROIEngine()
    roi_list = engine.compute(mock_results, task_frequencies=frequencies)

    # ── Per-task table ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"  {'TASK':<34} {'LABEL':<14} {'COST SAVED/YR':>14} {'ROI':>6}")
    print("=" * 75)
    for r in roi_list:
        print(
            f"  {r.task_name:<34} {r.label:<14} "
            f"${r.cost_saved_usd_yr:>12,.0f}  {r.roi_score:>5.1f}"
        )

    # ── Summary ────────────────────────────────────────────────────────
    s = engine.summary(roi_list)
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    print(f"  Total annual cost saving  : ${s['total_cost_saved_usd_yr']:>12,.0f}")
    print(f"  Total hours saved / year  : {s['total_hours_saved_yr']:>12,.0f} hrs")
    print(f"  Avg accuracy improvement  : {s['avg_accuracy_delta_pct']:>11.1f} pp")
    print(f"  Payback period (100% AI)  : {str(s['payback_months_at_100pct']):>11} months")
    print(f"  Top 3 tasks by ROI score  : {', '.join(s['top_3_tasks_by_roi'])}")

    # ── What-if simulation ─────────────────────────────────────────────
    whatif = engine.what_if_simulation(roi_list)
    print("\n" + "=" * 75)
    print("  WHAT-IF SIMULATION  (adoption scenarios)")
    print("=" * 75)
    print(whatif.to_string(index=False))

    # ── Save to DB ─────────────────────────────────────────────────────
    engine.save_to_db(roi_list)

    print("\nroi_engine.py smoke-test complete.")
    print(f"Results saved to: {DB_PATH}")