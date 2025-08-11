from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .analyzer import IncidentAnalyzer, Metrics


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _bar_chart(labels: List[str], values: List[float], title: str, rotation: int = 0) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4F46E5")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xticklabels(labels, rotation=rotation, ha="right")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _line_chart(labels: List[str], values: List[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, values, marker="o", color="#16A34A")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    return _fig_to_base64(fig)


def render_html(analyzer: IncidentAnalyzer, metrics: Metrics, template_dir: str, incidents_preview: int = 20) -> str:
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    sev_labels = list(metrics.incidents_by_severity.keys())
    sev_values = [metrics.incidents_by_severity[k] for k in sev_labels]
    svc_labels = list(metrics.incidents_by_service.keys())[:12]
    svc_values = [metrics.incidents_by_service[k] for k in svc_labels]
    weekly_labels = [w for w, _ in metrics.weekly_count]
    weekly_values = [c for _, c in metrics.weekly_count]

    charts = {
        "by_severity": _bar_chart(sev_labels, sev_values, "Incidents by severity"),
        "by_service": _bar_chart(svc_labels, svc_values, "Top services by incidents", rotation=45),
        "weekly": _line_chart(weekly_labels, weekly_values, "Incidents per week"),
    }

    similar = analyzer.find_similar_clusters()
    keywords = analyzer.top_keywords()
    suggestions = analyzer.suggested_root_causes()
    incidents = analyzer.to_dicts()

    return template.render(
        metrics=metrics,
        charts=charts,
        incidents=incidents[:incidents_preview],
        similar_clusters=similar,
        root_cause_suggestions=suggestions,
        keywords=keywords,
    )