"""
parser.py — Input handler for AI Workflow Optimizer
=====================================================
Accepts two input types:
  1. Free text  : user describes their business workflow in plain English
  2. Event logs : CSV / XES / XML (XES-format) from the data/ directory

Auto-discovery: place any supported file in data/ and the parser
will find it automatically — no hardcoded paths needed.

Supported file extensions
  .csv          — delimited event log (any separator, auto-detected)
  .xes          — IEEE XES process mining standard
  .xml          — XES file saved with .xml extension (e.g. BPI Challenge 2019)
  .xes.gz       — gzip-compressed XES

Outputs a unified list of WorkflowTask objects that feed into the
graph builder and classifier downstream.

Author  : AI Workflow Optimizer Project
Cost    : $0 — uses spaCy (local) + PM4Py (local), no API calls
Install : pip install spacy pm4py pandas
          python -m spacy download en_core_web_sm

Project layout expected
-----------------------
  project_root/
  ├── data/
  │   ├── BPI_Challenge_2019.xml      ← BPI 2019 XES (renamed .xml)
  │   ├── incident_management.csv
  │   └── insurance_claims.csv
  ├── src/
  │   └── parser.py                   ← this file
  └── ...
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import xml.etree.ElementTree as ET

import pandas as pd
import spacy
import pm4py
from pm4py.objects.log.obj import EventLog

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-level data directory
# Resolves to <project_root>/data/ regardless of where Python is invoked.
# Place all event log files here — the parser will find them automatically.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Supported file extensions
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv"   : "csv",
    ".xes"   : "xes",
    ".xml"   : "xml",    # XES files distributed as .xml (e.g. BPI 2019)
    ".xes.gz": "xes",
}


def _is_xes_xml(filepath: Path) -> bool:
    """
    Return True if a .xml file is actually XES format.
    Peeks at the root tag — XES files have <log ...> as root element.
    Falls back gracefully to False on any parse error.
    """
    try:
        for _, elem in ET.iterparse(str(filepath), events=("start",)):
            return elem.tag in ("log", "{http://www.xes-standard.org/}log")
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class WorkflowTask:
    """
    A single task node extracted from a workflow.

    Attributes
    ----------
    task_id   : unique identifier  e.g. "task_003"
    name      : short task label   e.g. "Review invoice"
    source    : where it came from — "text" | "csv" | "xes"
    frequency : how often it occurs (from event logs; defaults to 1)
    duration  : avg duration in minutes (from event logs; None if unknown)
    raw_text  : original sentence or log row for traceability
    metadata  : any extra key-value data (role, department, system, etc.)
    """
    task_id   : str
    name      : str
    source    : str
    frequency : int                    = 1
    duration  : Optional[float]        = None
    raw_text  : str                    = ""
    metadata  : dict                   = field(default_factory=dict)

    def __repr__(self) -> str:
        dur = f"{self.duration:.1f} min" if self.duration else "unknown"
        return (
            f"WorkflowTask(id={self.task_id!r}, name={self.name!r}, "
            f"source={self.source!r}, freq={self.frequency}, duration={dur})"
        )


# ---------------------------------------------------------------------------
# Text parser — spaCy-based
# ---------------------------------------------------------------------------
class TextParser:
    """
    Extracts workflow tasks from free-form English descriptions.

    Strategy
    --------
    1. Split input into sentences.
    2. For each sentence, extract the dominant verb-object phrase.
       This turns "The team reviews incoming invoices" → "review invoices".
    3. Filter out noise (conjunctions, filler words, very short phrases).
    4. Deduplicate by normalised lemma form.

    The result is a clean, human-readable task name per sentence that is
    short enough to be a graph node label.
    """

    # Words that appear as verbs but carry no workflow meaning
    _STOP_VERBS = {
        "be", "have", "do", "get", "make", "go", "come",
        "know", "think", "say", "tell", "use", "include",
    }

    def __init__(self, model: str = "en_core_web_sm") -> None:
        logger.info("Loading spaCy model '%s' ...", model)
        try:
            self._nlp = spacy.load(model)
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found.\n"
                f"Run:  python -m spacy download {model}"
            )
        logger.info("spaCy model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self, text: str) -> list[WorkflowTask]:
        """
        Parse a block of workflow text and return a list of WorkflowTask.

        Parameters
        ----------
        text : plain-English description of a business workflow

        Returns
        -------
        list[WorkflowTask]
        """
        if not text or not text.strip():
            logger.warning("Empty text received — returning no tasks.")
            return []

        text = self._preprocess(text)
        doc  = self._nlp(text)

        tasks     = []
        seen      = set()          # dedup by normalised name
        task_idx  = 1

        for sent in doc.sents:
            task_name = self._extract_task_name(sent)
            if not task_name:
                continue

            # Normalise for dedup: lowercase, collapse spaces
            norm = re.sub(r"\s+", " ", task_name.lower()).strip()
            if norm in seen or len(norm) < 4:
                continue
            seen.add(norm)

            tasks.append(WorkflowTask(
                task_id  = f"task_{task_idx:03d}",
                name     = task_name.capitalize(),
                source   = "text",
                raw_text = sent.text.strip(),
            ))
            task_idx += 1
            logger.debug("Extracted task: %s", task_name)

        logger.info("Text parser found %d task(s).", len(tasks))
        return tasks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(text: str) -> str:
        """Normalise whitespace and fix common formatting issues."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Ensure bullet-point lines end with a period so spaCy sees sentences
        text = re.sub(r"(?m)^[-•*]\s*(.+?)$", r"\1.", text)
        return text.strip()

    def _extract_task_name(self, sent) -> str | None:
        """
        Build a short task label from a spaCy Span (sentence).

        Tries, in order:
          (a) ROOT verb + direct object
          (b) ROOT verb + prepositional complement
          (c) ROOT verb alone (if meaningful)
          (d) First noun chunk if no usable verb found
        """
        root = sent.root

        # (a) verb + direct object
        if root.pos_ == "VERB" and root.lemma_ not in self._STOP_VERBS:
            dobjs = [t for t in root.children if t.dep_ == "dobj"]
            if dobjs:
                obj_phrase = self._subtree_text(dobjs[0], max_tokens=4)
                return f"{root.lemma_} {obj_phrase}"

            # (b) verb + prep phrase  e.g. "respond to customer"
            preps = [t for t in root.children if t.dep_ == "prep"]
            for prep in preps:
                pobjs = [t for t in prep.children if t.dep_ == "pobj"]
                if pobjs:
                    obj_phrase = self._subtree_text(pobjs[0], max_tokens=3)
                    return f"{root.lemma_} {prep.text} {obj_phrase}"

            # (c) meaningful verb alone
            if len(root.lemma_) > 3:
                return root.lemma_

        # (d) fallback: first noun chunk
        chunks = list(sent.noun_chunks)
        if chunks:
            return chunks[0].text.lower()

        return None

    @staticmethod
    def _subtree_text(token, max_tokens: int = 4) -> str:
        """Return a compact string from a token's subtree, capped at max_tokens."""
        tokens = [t.text.lower() for t in token.subtree
                  if not t.is_punct and not t.is_space]
        return " ".join(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# CSV / XES parser — PM4Py-based
# ---------------------------------------------------------------------------
class EventLogParser:
    """
    Extracts workflow tasks from structured process event logs.

    Supported formats
    -----------------
    - CSV  : any delimiter, auto-detected
    - XES  : standard process-mining exchange format

    Required columns (CSV)
    ----------------------
    The parser auto-detects common column name variants:

    | Concept       | Accepted column names                        |
    |---------------|----------------------------------------------|
    | Activity name | activity, concept:name, task, action, event  |
    | Case ID       | case_id, case:concept:name, caseid, id        |
    | Timestamp     | timestamp, time:timestamp, start_time, date  |

    Optional columns
    ----------------
    - duration / duration_minutes / elapsed  → avg duration per task
    - role / resource / department           → stored in metadata
    """

    # Column name aliases — add more here if your dataset uses other names
    _ACTIVITY_ALIASES  = ["activity", "concept:name", "task", "action", "event", "activity_name"]
    _CASE_ALIASES      = ["case_id", "case:concept:name", "caseid", "case", "id", "case_concept_name"]
    _TIMESTAMP_ALIASES = ["timestamp", "time:timestamp", "start_time", "date", "datetime", "time"]
    _DURATION_ALIASES  = ["duration", "duration_minutes", "elapsed", "time_taken"]
    _ROLE_ALIASES      = ["role", "resource", "department", "team", "actor", "org:resource"]

    def parse_csv(self, filepath: str | Path) -> list[WorkflowTask]:
        """
        Load a CSV event log and extract unique workflow tasks.

        Parameters
        ----------
        filepath : path to the CSV file

        Returns
        -------
        list[WorkflowTask]
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        logger.info("Reading CSV: %s", filepath)
        df = pd.read_csv(filepath, sep=None, engine="python")
        df.columns = [c.strip().lower() for c in df.columns]

        logger.info("CSV columns detected: %s", list(df.columns))

        activity_col  = self._find_col(df, self._ACTIVITY_ALIASES,  "activity")
        duration_col  = self._find_col(df, self._DURATION_ALIASES,  None, required=False)
        role_col      = self._find_col(df, self._ROLE_ALIASES,       None, required=False)

        return self._build_tasks(df, activity_col, duration_col, role_col, source="csv")

    def parse_xes(self, filepath: str | Path) -> list[WorkflowTask]:
        """
        Load an XES event log via PM4Py and extract unique workflow tasks.

        Parameters
        ----------
        filepath : path to the .xes file

        Returns
        -------
        list[WorkflowTask]
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"XES/XML file not found: {filepath}")

        # PM4Py requires a .xes extension. BPI Challenge files are often
        # distributed as .xml — we copy to a temp .xes file transparently.
        target = filepath
        _tmp: Path | None = None
        if filepath.suffix.lower() == ".xml":
            import shutil, tempfile
            _tmp = Path(tempfile.mktemp(suffix=".xes"))
            shutil.copy2(filepath, _tmp)
            target = _tmp
            logger.info(
                "BPI/XML file detected — copied to temp .xes for PM4Py: %s", _tmp
            )

        try:
            logger.info("Reading XES: %s", target)
            log: EventLog = pm4py.read_xes(str(target))
        finally:
            if _tmp and _tmp.exists():
                _tmp.unlink()
                logger.debug("Removed temp file: %s", _tmp)

        df = pm4py.convert_to_dataframe(log)
        df.columns = [c.strip().lower() for c in df.columns]
        logger.info(
            "XES loaded — %d rows, %d columns: %s",
            len(df), len(df.columns), list(df.columns)
        )

        activity_col = self._find_col(df, self._ACTIVITY_ALIASES, "activity")
        duration_col = self._find_col(df, self._DURATION_ALIASES, None, required=False)
        role_col     = self._find_col(df, self._ROLE_ALIASES,      None, required=False)

        return self._build_tasks(df, activity_col, duration_col, role_col, source="xes")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_tasks(
        self,
        df          : pd.DataFrame,
        activity_col: str,
        duration_col: str | None,
        role_col    : str | None,
        source      : str,
    ) -> list[WorkflowTask]:
        """Aggregate rows by activity and produce WorkflowTask objects."""

        tasks    = []
        task_idx = 1

        # Group by activity to get frequency and avg duration
        grouped = df.groupby(activity_col)

        for activity_name, group in grouped:
            activity_name = str(activity_name).strip()
            if not activity_name or activity_name.lower() in ("nan", "none", ""):
                continue

            freq = len(group)

            avg_duration = None
            if duration_col:
                durations = pd.to_numeric(group[duration_col], errors="coerce")
                if durations.notna().any():
                    avg_duration = round(float(durations.mean()), 2)

            meta = {}
            if role_col:
                roles = group[role_col].dropna().unique().tolist()
                if roles:
                    meta["roles"] = [str(r) for r in roles[:5]]  # cap at 5

            tasks.append(WorkflowTask(
                task_id  = f"task_{task_idx:03d}",
                name     = activity_name,
                source   = source,
                frequency= freq,
                duration = avg_duration,
                raw_text = f"Observed {freq} time(s) in event log.",
                metadata = meta,
            ))
            task_idx += 1

        # Sort by frequency descending (most common tasks first)
        tasks.sort(key=lambda t: t.frequency, reverse=True)

        logger.info("Event log parser found %d unique task(s).", len(tasks))
        return tasks

    @staticmethod
    def _find_col(
        df      : pd.DataFrame,
        aliases : list[str],
        default : str | None,
        required: bool = True,
    ) -> str | None:
        """Find the first matching column name from a list of aliases."""
        for alias in aliases:
            if alias in df.columns:
                logger.debug("Matched column alias '%s'.", alias)
                return alias
        if required:
            raise ValueError(
                f"Could not find a required column. "
                f"Expected one of: {aliases}.\n"
                f"Available columns: {list(df.columns)}"
            )
        return default if default in df.columns else None


# ---------------------------------------------------------------------------
# Unified facade — this is what all other modules import
# ---------------------------------------------------------------------------
class WorkflowParser:
    """
    Single entry-point for all input types.

    Auto-discovery
    --------------
    Place any supported file in the  data/  directory at project root
    and call  parse_data_dir()  — no hardcoded paths needed.

    Manual usage
    ------------
        parser = WorkflowParser()

        # From free text
        tasks = parser.parse_text("First, the team receives an email...")

        # Specific file (CSV, XES, or .xml XES)
        tasks = parser.parse_file("data/BPI_Challenge_2019.xml")

        # Auto-discover and parse ALL files in data/
        tasks = parser.parse_data_dir()

        # Parse one specific file inside data/
        tasks = parser.parse_data_dir("BPI_Challenge_2019.xml")

        # Mix text + file
        tasks = parser.parse_auto(text="...", filepath="data/log.csv")
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._text_parser     : TextParser | None = None   # lazy-loaded
        self._event_log_parser: EventLogParser     = EventLogParser()
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------
    def parse_text(self, text: str) -> list[WorkflowTask]:
        """Parse a plain-English workflow description."""
        if self._text_parser is None:
            self._text_parser = TextParser()
        return self._text_parser.parse(text)

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------
    def parse_file(self, filepath: str | Path) -> list[WorkflowTask]:
        """
        Parse a single event log file.
        Supports: .csv  .xes  .xml (XES format)  .xes.gz
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"File not found: {filepath}\n"
                f"Tip: place your data files in  {self.data_dir}"
            )

        ext = filepath.suffix.lower()

        if ext == ".csv":
            return self._event_log_parser.parse_csv(filepath)
        elif ext in (".xes", ".xes.gz"):
            return self._event_log_parser.parse_xes(filepath)
        elif ext == ".xml":
            if _is_xes_xml(filepath):
                logger.info("Detected XES-format XML: %s", filepath.name)
                return self._event_log_parser.parse_xes(filepath)
            else:
                raise ValueError(
                    f"'{filepath.name}' is XML but not XES format (root tag != <log>).\n"
                    f"If this is a citation/metadata file, use the actual event log file."
                )
        else:
            raise ValueError(
                f"Unsupported format: '{ext}'. "
                f"Supported: .csv  .xes  .xml (XES)  .xes.gz"
            )

    # ------------------------------------------------------------------
    # Data directory — auto-discovery
    # ------------------------------------------------------------------
    def discover_data_files(self) -> list[Path]:
        """
        Scan self.data_dir and return all supported event log files found.
        """
        if not self.data_dir.exists():
            logger.warning(
                "data/ directory not found at: %s — create it and "
                "place your event log files there.", self.data_dir
            )
            return []

        found: list[Path] = []
        for ext in _SUPPORTED_EXTENSIONS:
            found.extend(self.data_dir.glob(f"*{ext}"))
        found = sorted(set(found))

        if found:
            logger.info(
                "Auto-discovered %d file(s) in %s:", len(found), self.data_dir
            )
            for f in found:
                logger.info("  [%s]  %s", f.suffix.lstrip(".").upper(), f.name)
        else:
            logger.warning(
                "No supported files found in %s.\n"
                "Supported: %s", self.data_dir, list(_SUPPORTED_EXTENSIONS.keys())
            )
        return found

    def parse_data_dir(self, filename: str | None = None) -> list[WorkflowTask]:
        """
        Parse files from the data/ directory.

        Parameters
        ----------
        filename : optional specific filename inside data/ to parse.
                   If None, ALL supported files are parsed and merged.

        Examples
        --------
            tasks = parser.parse_data_dir("BPI_Challenge_2019.xml")
            tasks = parser.parse_data_dir()   # parse everything
        """
        if filename:
            return self.parse_file(self.data_dir / filename)

        files = self.discover_data_files()
        if not files:
            raise FileNotFoundError(
                f"No supported event log files in: {self.data_dir}\n"
                f"Add .csv / .xes / .xml files to that directory."
            )

        all_tasks: list[WorkflowTask] = []
        for f in files:
            try:
                tasks = self.parse_file(f)
                logger.info("  Parsed '%s' → %d tasks", f.name, len(tasks))
                all_tasks.extend(tasks)
            except Exception as exc:
                logger.warning("  Skipped '%s': %s", f.name, exc)

        for i, task in enumerate(all_tasks, start=1):
            task.task_id = f"task_{i:03d}"

        logger.info("Total tasks after merging: %d", len(all_tasks))
        return all_tasks

    # ------------------------------------------------------------------
    # Convenience: text + file combined
    # ------------------------------------------------------------------
    def parse_auto(
        self,
        text    : str | None        = None,
        filepath: str | Path | None = None,
    ) -> list[WorkflowTask]:
        """Pass text, a filepath, or both — merged and re-indexed."""
        results: list[WorkflowTask] = []
        if text:
            results.extend(self.parse_text(text))
        if filepath:
            results.extend(self.parse_file(filepath))
        if not results:
            raise ValueError("Provide at least one of: text or filepath.")
        for i, task in enumerate(results, start=1):
            task.task_id = f"task_{i:03d}"
        return results


# ---------------------------------------------------------------------------
# Quick smoke-test  (run: python src/parser.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    parser = WorkflowParser()

    # ── 1. Text demo ────────────────────────────────────────────────────
    sample_text = """
    First, the customer sends an email inquiry to the support team.
    A support agent reads the email and assigns it to the correct department.
    The department manager reviews the case and approves the response.
    A junior analyst drafts the reply using a template.
    The manager approves the draft before it is sent.
    Finally, the system logs the interaction in the CRM database.
    """
    print("\n" + "=" * 60)
    print("  1. TEXT PARSING DEMO")
    print("=" * 60)
    for t in parser.parse_text(sample_text):
        print(t)

    # ── 2. CSV demo (auto-generated) ─────────────────────────────────────
    sample_csv = (
        "case_id,activity,timestamp,duration,role\n"
        "1,Receive email,2024-01-01 09:00,2,Support Agent\n"
        "1,Assign ticket,2024-01-01 09:02,1,Support Agent\n"
        "1,Review case,2024-01-01 09:10,5,Manager\n"
        "2,Receive email,2024-01-02 10:00,2,Support Agent\n"
        "2,Draft reply,2024-01-02 10:05,10,Analyst\n"
        "2,Approve reply,2024-01-02 10:20,3,Manager\n"
        "2,Send reply,2024-01-02 10:25,1,Support Agent\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp:
        tmp.write(sample_csv)
        tmp_path = Path(tmp.name)

    print("\n" + "=" * 60)
    print("  2. CSV PARSING DEMO")
    print("=" * 60)
    for t in parser.parse_file(tmp_path):
        print(t)
    tmp_path.unlink()

    # ── 3. data/ directory discovery ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  3. DATA DIRECTORY SCAN")
    print("=" * 60)
    files = parser.discover_data_files()
    if files:
        print(f"Found {len(files)} file(s) ready to parse:")
        for f in files:
            print(f"  → {f.name}")
        print("\nTo parse BPI 2019 XES data run:")
        print("  tasks = parser.parse_data_dir('BPI_Challenge_2019.xml')")
    else:
        print(f"Place your .xml / .xes / .csv files in:  {parser.data_dir}")

    print("\nparser.py smoke-test complete.")