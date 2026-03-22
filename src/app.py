"""
app.py — AI Workflow Optimization Engine Dashboard
====================================================
Streamlit dashboard that ties together:
  parser.py      → input handling (text + XES/CSV)
  classifier.py  → AI opportunity detection (local BART)
  roi_engine.py  → ROI estimation + what-if simulation

Sections
--------
  1. Sidebar       — inputs, config, run button
  2. Summary cards — total saving, hours freed, payback, tasks analysed
  3. Heatmap       — AI opportunity heatmap by label + confidence
  4. ROI bar chart — cost saved per task (ranked)
  5. Workflow graph— NetworkX DAG visualised via PyVis
  6. What-if sim   — adoption scenario slider
  7. Data table    — full results + CSV export

Deploy free: streamlit run app.py
             OR push to GitHub → share on Streamlit Community Cloud

Author  : AI Workflow Optimizer Project
Cost    : $0 — Streamlit free tier + local BART model
Install : pip install streamlit plotly pyvis networkx pandas numpy
          pip install transformers torch spacy pm4py
          python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import sys
import tempfile
import sqlite3
from pathlib import Path
from io import StringIO

import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Path setup — works whether run from project root or src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser     import WorkflowParser
from classifier import WorkflowClassifier, AILabel
from roi_engine import ROIEngine, ROIConfig

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "AI Workflow Optimizer",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Theme colours per label
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    "automatable": "#1D9E75",   # teal
    "augmentable": "#378ADD",   # blue
    "non_ai"     : "#888780",   # gray
}
LABEL_ICONS = {
    "automatable": "🤖",
    "augmentable": "🤝",
    "non_ai"     : "🧑",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _badge(label: str) -> str:
    icon = LABEL_ICONS.get(label, "")
    return f"{icon} {label.replace('_', ' ').title()}"


@st.cache_resource(show_spinner="Loading BART model (first run ~30s)...")
def _load_classifier() -> WorkflowClassifier:
    return WorkflowClassifier(use_local=True)


@st.cache_data(show_spinner="Parsing workflow...")
def _parse_text(text: str) -> list:
    return WorkflowParser().parse_text(text)


@st.cache_data(show_spinner="Parsing event log...")
def _parse_file(file_bytes: bytes, suffix: str) -> list:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    tasks = WorkflowParser().parse_file(tmp_path)
    tmp_path.unlink()
    return tasks


def _run_pipeline(
    tasks      : list,
    config     : ROIConfig,
    freq_scale : float,
) -> tuple[list, list, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Run classifier → ROI engine → return all results."""
    classifier = _load_classifier()

    with st.spinner("Classifying tasks with BART..."):
        results = classifier.classify_all(tasks, save_to_db=False)

    frequencies = {
        t.task_id: max(1, int(t.frequency * freq_scale))
        for t in tasks
    }
    durations = {
        t.task_id: (t.duration / 60.0) if t.duration else None
        for t in tasks
    }
    durations = {k: v for k, v in durations.items() if v}

    engine   = ROIEngine(config=config)
    roi_list = engine.compute(results, frequencies, durations or None)
    roi_df   = engine.to_dataframe(roi_list)
    cls_df   = classifier.to_dataframe(results)
    summary  = engine.summary(roi_list)
    whatif   = engine.what_if_simulation(roi_list)

    return results, roi_list, roi_df, cls_df, summary, whatif


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _heatmap(cls_df: pd.DataFrame) -> go.Figure:
    """AI opportunity heatmap — label × confidence bubble chart."""
    df = cls_df.copy()
    df["label_display"] = df["label"].map(_badge)
    df["size"] = (df["confidence"] * 60).clip(lower=10)

    fig = px.scatter(
        df,
        x          = "confidence",
        y          = "label_display",
        size       = "size",
        color      = "label",
        color_discrete_map = {
            k: LABEL_COLORS[k] for k in LABEL_COLORS
        },
        hover_name = "task_name",
        hover_data = {"confidence": ":.2f", "size": False, "label": False},
        title      = "AI opportunity heatmap",
        labels     = {"confidence": "Classifier confidence", "label_display": ""},
        height     = 380,
    )
    fig.update_layout(
        showlegend   = False,
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor= "rgba(0,0,0,0)",
        font_color   = "#444441",
        title_font_size = 15,
        xaxis = dict(range=[0, 1.05], gridcolor="#E8E6DF"),
        yaxis = dict(gridcolor="#E8E6DF"),
        margin = dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _roi_bar(roi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart — cost saved per task."""
    df = roi_df[roi_df["cost_saved_usd_yr"] > 0].copy()
    df = df.sort_values("cost_saved_usd_yr", ascending=True).tail(15)
    df["color"] = df["label"].map(LABEL_COLORS)
    df["label_display"] = df["cost_saved_usd_yr"].apply(
        lambda x: f"${x:,.0f}"
    )

    fig = go.Figure(go.Bar(
        x           = df["cost_saved_usd_yr"],
        y           = df["task_name"],
        orientation = "h",
        marker_color= df["color"],
        text        = df["label_display"],
        textposition= "outside",
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Cost saved: $%{x:,.0f}/yr<extra></extra>"
        ),
    ))
    fig.update_layout(
        title        = "Annual cost saving per task (USD)",
        title_font_size = 15,
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor= "rgba(0,0,0,0)",
        font_color   = "#444441",
        xaxis = dict(
            title      = "Annual saving (USD)",
            gridcolor  = "#E8E6DF",
            tickprefix = "$",
            tickformat = ",.0f",
        ),
        yaxis  = dict(title=""),
        height = max(350, len(df) * 36),
        margin = dict(l=20, r=120, t=50, b=20),
    )
    return fig


def _whatif_chart(whatif_df: pd.DataFrame) -> go.Figure:
    """Line chart — adoption scenario vs annual saving."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = whatif_df["adoption_pct"],
        y    = whatif_df["annual_saving_usd"],
        mode = "lines+markers",
        name = "Annual saving",
        line = dict(color=LABEL_COLORS["automatable"], width=3),
        marker = dict(size=9),
        hovertemplate = "Adoption: %{x}<br>Saving: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x    = whatif_df["adoption_pct"],
        y    = whatif_df["hours_saved_yr"],
        name = "Hours saved/yr",
        marker_color = LABEL_COLORS["augmentable"],
        opacity      = 0.35,
        yaxis        = "y2",
        hovertemplate = "Adoption: %{x}<br>Hours: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title        = "What-if simulation — AI adoption scenarios",
        title_font_size = 15,
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor= "rgba(0,0,0,0)",
        font_color   = "#444441",
        height       = 380,
        legend       = dict(orientation="h", y=1.12),
        xaxis        = dict(title="Adoption level", gridcolor="#E8E6DF"),
        yaxis        = dict(
            title      = "Annual saving (USD)",
            tickprefix = "$",
            tickformat = ",.0f",
            gridcolor  = "#E8E6DF",
        ),
        yaxis2 = dict(
            title    = "Hours saved / yr",
            overlaying = "y",
            side       = "right",
            showgrid   = False,
        ),
        margin = dict(l=20, r=80, t=70, b=20),
    )
    return fig


def _workflow_graph_html(tasks: list, cls_df: pd.DataFrame) -> str:
    """Build a PyVis workflow graph and return HTML string."""
    G   = nx.DiGraph()
    net = Network(height="480px", width="100%", directed=True, bgcolor="#ffffff")
    net.set_options("""
    {
      "physics": {"barnesHut": {"gravitationalConstant": -8000}},
      "edges":   {"arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
                  "color":  {"color": "#B4B2A9"},
                  "width":  1.5},
      "nodes":   {"font": {"size": 13}}
    }
    """)

    label_map = {}
    if not cls_df.empty:
        label_map = dict(zip(cls_df["task_name"], cls_df["label"]))

    for i, task in enumerate(tasks):
        lbl   = label_map.get(task.name, "non_ai")
        color = LABEL_COLORS.get(lbl, "#888780")
        net.add_node(
            task.task_id,
            label = task.name,
            color = color,
            title = (
                f"<b>{task.name}</b><br>"
                f"Label: {lbl}<br>"
                f"Frequency: {task.frequency:,}"
            ),
            size  = min(20 + np.log1p(task.frequency) * 2, 45),
        )
        G.add_node(task.task_id)

    # Connect tasks sequentially (simple linear workflow assumption)
    for i in range(len(tasks) - 1):
        net.add_edge(tasks[i].task_id, tasks[i + 1].task_id)
        G.add_edge(tasks[i].task_id, tasks[i + 1].task_id)

    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w"
    ) as tmp:
        net.save_graph(tmp.name)
        html = Path(tmp.name).read_text()
        Path(tmp.name).unlink()

    return html


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar() -> tuple[list | None, ROIConfig, float, bool]:
    """Render sidebar and return (tasks, config, freq_scale, run_clicked)."""

    st.sidebar.image(
        "https://img.shields.io/badge/AI%20Workflow%20Optimizer-HBS%20Case-purple?style=for-the-badge",
        use_column_width=True,
    )
    st.sidebar.markdown("---")

    # ── Input mode ────────────────────────────────────────────────────
    st.sidebar.subheader("📥 Input")
    mode = st.sidebar.radio(
        "Workflow source",
        ["Upload event log (.xes / .xml / .csv)", "Paste text description"],
        label_visibility="collapsed",
    )

    tasks = None

    if mode == "Upload event log (.xes / .xml / .csv)":
        uploaded = st.sidebar.file_uploader(
            "Upload event log",
            type=["xes", "xml", "csv"],
            help="Supports BPI Challenge .xes/.xml files and any PM4Py-compatible CSV",
        )
        if uploaded:
            with st.spinner("Parsing event log..."):
                try:
                    tasks = _parse_file(
                        uploaded.read(),
                        suffix=Path(uploaded.name).suffix,
                    )
                    st.sidebar.success(f"✅ {len(tasks)} unique tasks found")
                except Exception as e:
                    st.sidebar.error(f"Parse error: {e}")

    else:
        default_text = (
            "First, the procurement team creates a purchase order in SAP.\n"
            "The system automatically routes the order for manager approval.\n"
            "A manager reviews and approves the purchase order.\n"
            "The vendor receives the order and creates an invoice.\n"
            "The finance team validates the invoice against the goods receipt.\n"
            "The system clears the invoice and schedules payment.\n"
            "The payment is sent to the vendor and logged in the system."
        )
        text = st.sidebar.text_area(
            "Workflow description",
            value   = default_text,
            height  = 200,
            help    = "Describe your business workflow in plain English",
        )
        if text.strip():
            with st.spinner("Parsing text..."):
                try:
                    tasks = _parse_text(text)
                    st.sidebar.success(f"✅ {len(tasks)} tasks extracted")
                except Exception as e:
                    st.sidebar.error(f"Parse error: {e}")

    st.sidebar.markdown("---")

    # ── ROI Config ────────────────────────────────────────────────────
    st.sidebar.subheader("⚙️ ROI assumptions")

    hourly_rate = st.sidebar.number_input(
        "Hourly rate (USD)",
        min_value = 10.0,
        max_value = 500.0,
        value     = 45.0,
        step      = 5.0,
        help      = "Fully-loaded hourly cost per knowledge worker",
    )
    impl_cost = st.sidebar.number_input(
        "Implementation cost (USD)",
        min_value = 1_000.0,
        max_value = 1_000_000.0,
        value     = 50_000.0,
        step      = 5_000.0,
        help      = "One-time cost to implement AI across the workflow",
    )
    freq_scale = st.sidebar.slider(
        "Frequency scale",
        min_value = 0.1,
        max_value = 2.0,
        value     = 1.0,
        step      = 0.1,
        help      = "Scale event frequencies up/down for scenario testing",
    )

    config = ROIConfig(
        hourly_rate_usd         = hourly_rate,
        implementation_cost_usd = impl_cost,
    )

    st.sidebar.markdown("---")

    run = st.sidebar.button(
        "🚀 Run Analysis",
        use_container_width = True,
        type                = "primary",
        disabled            = (tasks is None or len(tasks) == 0),
    )

    return tasks, config, freq_scale, run


# ---------------------------------------------------------------------------
# Main dashboard sections
# ---------------------------------------------------------------------------
def _render_summary_cards(summary: dict) -> None:
    """4 KPI cards across the top."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "💰 Annual cost saving",
            f"${summary.get('total_cost_saved_usd_yr', 0):,.0f}",
            help="Total USD saved per year if all automatable/augmentable tasks get AI",
        )
    with c2:
        st.metric(
            "⏱️ Hours freed / year",
            f"{summary.get('total_hours_saved_yr', 0):,.0f} hrs",
            help="Total human hours per year returned by AI adoption",
        )
    with c3:
        pb = summary.get("payback_months_at_100pct")
        st.metric(
            "📈 Payback period",
            f"{pb} months" if pb else "N/A",
            help="Months to recover implementation cost at 100% AI adoption",
        )
    with c4:
        auto_pct = round(
            summary.get("automatable_count", 0) /
            max(summary.get("total_tasks", 1), 1) * 100, 1
        )
        aug_pct  = round(
            summary.get("augmentable_count", 0) /
            max(summary.get("total_tasks", 1), 1) * 100, 1
        )
        st.metric(
            "🤖 AI-ready tasks",
            f"{auto_pct + aug_pct:.0f}%",
            help=f"Automatable: {auto_pct}%  |  Augmentable: {aug_pct}%",
        )


def _render_heatmap_and_bar(cls_df: pd.DataFrame, roi_df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_heatmap(cls_df), use_container_width=True)
    with col2:
        st.plotly_chart(_roi_bar(roi_df), use_container_width=True)


def _render_graph(tasks: list, cls_df: pd.DataFrame) -> None:
    st.subheader("🔗 Workflow graph")
    st.caption(
        "Node size = task frequency  •  "
        "🟢 Automatable  •  🔵 Augmentable  •  ⚫ Non-AI"
    )
    html = _workflow_graph_html(tasks, cls_df)
    st.components.v1.html(html, height=500, scrolling=False)


def _render_whatif(whatif_df: pd.DataFrame, summary: dict) -> None:
    st.subheader("🔮 What-if simulation")

    adoption_labels = whatif_df["adoption_pct"].tolist()
    selected = st.select_slider(
        "AI adoption level",
        options = adoption_labels,
        value   = "100%",
        help    = "Drag to see projected ROI at different AI adoption levels",
    )

    row = whatif_df[whatif_df["adoption_pct"] == selected].iloc[0]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tasks enabled",    f"{int(row['tasks_enabled'])}")
    s2.metric("Annual saving",    f"${row['annual_saving_usd']:,.0f}")
    s3.metric("Hours saved / yr", f"{row['hours_saved_yr']:,.0f}")
    pb = row["payback_months"]
    s4.metric("Payback",          f"{pb} months" if pd.notna(pb) else "N/A")

    st.plotly_chart(_whatif_chart(whatif_df), use_container_width=True)


def _render_data_table(roi_df: pd.DataFrame, cls_df: pd.DataFrame) -> None:
    st.subheader("📋 Full results")

    tab1, tab2 = st.tabs(["ROI breakdown", "Classification details"])

    with tab1:
        display_cols = [
            "task_name", "label", "frequency_per_yr",
            "cost_saved_usd_yr", "time_saved_hrs_yr",
            "accuracy_delta_pct", "roi_score", "tool_suggestion",
        ]
        available = [c for c in display_cols if c in roi_df.columns]
        st.dataframe(
            roi_df[available].style.format({
                "cost_saved_usd_yr" : "${:,.0f}",
                "time_saved_hrs_yr" : "{:,.1f}",
                "roi_score"         : "{:.1f}",
                "confidence"        : "{:.2f}",
            }),
            use_container_width = True,
            height              = 350,
        )
        csv = roi_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download ROI report (CSV)",
            data      = csv,
            file_name = "ai_workflow_roi_report.csv",
            mime      = "text/csv",
        )

    with tab2:
        display_cls = [
            "task_name", "label", "confidence",
            "score_automatable", "score_augmentable", "score_non_ai",
            "reasoning", "tool_suggestion",
        ]
        available_cls = [c for c in display_cls if c in cls_df.columns]
        st.dataframe(
            cls_df[available_cls].style.format({
                "confidence"       : "{:.2f}",
                "score_automatable": "{:.3f}",
                "score_augmentable": "{:.3f}",
                "score_non_ai"     : "{:.3f}",
            }),
            use_container_width = True,
            height              = 350,
        )
        csv2 = cls_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download classification details (CSV)",
            data      = csv2,
            file_name = "ai_workflow_classifications.csv",
            mime      = "text/csv",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Header
    st.title("🧠 AI Workflow Optimization Engine")
    st.caption(
        "Harvard Business School case — Generative AI in Business Workflows  •  "
        "Built with BART, PM4Py, Streamlit  •  $0 cost"
    )
    st.markdown("---")

    # Sidebar
    tasks, config, freq_scale, run_clicked = _sidebar()

    # Session state — persist results across reruns
    if "results_ready" not in st.session_state:
        st.session_state.results_ready = False

    if run_clicked and tasks:
        with st.spinner("Running full pipeline..."):
            try:
                (
                    results, roi_list,
                    roi_df, cls_df,
                    summary, whatif_df,
                ) = _run_pipeline(tasks, config, freq_scale)

                st.session_state.results_ready = True
                st.session_state.results       = results
                st.session_state.roi_list      = roi_list
                st.session_state.roi_df        = roi_df
                st.session_state.cls_df        = cls_df
                st.session_state.summary       = summary
                st.session_state.whatif_df     = whatif_df
                st.session_state.tasks         = tasks

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

    # ── Render results ────────────────────────────────────────────────
    if st.session_state.get("results_ready"):
        roi_df   = st.session_state.roi_df
        cls_df   = st.session_state.cls_df
        summary  = st.session_state.summary
        whatif_df= st.session_state.whatif_df
        tasks_   = st.session_state.tasks

        # 1. KPI cards
        _render_summary_cards(summary)
        st.markdown("---")

        # 2. Heatmap + ROI bar chart
        _render_heatmap_and_bar(cls_df, roi_df)
        st.markdown("---")

        # 3. Workflow graph
        _render_graph(tasks_, cls_df)
        st.markdown("---")

        # 4. What-if simulation
        _render_whatif(whatif_df, summary)
        st.markdown("---")

        # 5. Data table + exports
        _render_data_table(roi_df, cls_df)

    else:
        # Landing state — show instructions
        st.info(
            "👈 **Upload your event log or paste a workflow description** "
            "in the sidebar, then click **Run Analysis**."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Step 1 — Load data**
            Upload your BPI 2019 `.xml` file
            or any PM4Py-compatible `.csv`
            or paste a text description.
            """)
        with col2:
            st.markdown("""
            **Step 2 — Configure**
            Adjust hourly rate and
            implementation cost to match
            your organisation.
            """)
        with col3:
            st.markdown("""
            **Step 3 — Analyse**
            Click Run Analysis to get
            AI opportunity labels, ROI
            estimates and what-if scenarios.
            """)


if __name__ == "__main__":
    main()