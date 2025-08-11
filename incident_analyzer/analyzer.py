from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dateutil import parser as dateparser


CANONICAL_COLS = {
    "id": ["id", "incident_id", "incidentId", "incident-id"],
    "title": ["title", "name", "summary"],
    "description": ["description", "desc", "details"],
    "severity": ["severity", "sev", "level"],
    "service": ["service", "service_name", "system", "component"],
    "status": ["status", "state"],
    "start_time": [
        "start_time",
        "start",
        "started_at",
        "startTime",
        "startedAt",
        "created_at",
        "createdAt",
        "opened_at",
    ],
    "ack_time": [
        "ack_time",
        "acknowledged_at",
        "acknowledgedAt",
        "first_ack_time",
        "firstAckTime",
    ],
    "end_time": [
        "end_time",
        "end",
        "ended_at",
        "endTime",
        "endedAt",
        "resolved_at",
        "resolvedAt",
        "closed_at",
    ],
    "tags": ["tags", "labels"],
    "cause": ["cause", "root_cause", "rootCause"],
    "impact_minutes": ["impact_minutes", "impactMinutes", "impact_mins"],
}


@dataclass
class Metrics:
    total_incidents: int
    mttr_minutes: Optional[float]
    mtta_minutes: Optional[float]
    incidents_by_severity: Dict[str, int]
    incidents_by_service: Dict[str, int]
    weekly_count: List[Tuple[str, int]]
    p50_mttr: Optional[float]
    p90_mttr: Optional[float]
    p50_mtta: Optional[float]
    p90_mtta: Optional[float]


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            # Heuristic: > 1e12 likely ms
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None
    try:
        dt = dateparser.parse(str(value))
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _normalize_severity(sev: Any) -> str:
    if sev is None:
        return "unknown"
    s = str(sev).strip().lower()
    mapping = {
        "1": "sev1",
        "2": "sev2",
        "3": "sev3",
        "4": "sev4",
        "5": "sev5",
        "p0": "sev1",
        "p1": "sev2",
        "p2": "sev3",
        "p3": "sev4",
        "p4": "sev5",
        "critical": "sev1",
        "high": "sev2",
        "medium": "sev3",
        "low": "sev4",
    }
    return mapping.get(s, s)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def _percentile(values: List[float], p: float) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    vals.sort()
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals[int(k)])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


class IncidentAnalyzer:
    def __init__(self) -> None:
        self.original_rows: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []

    def load(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "incidents" in data:
                data = data["incidents"]
            if not isinstance(data, list):
                raise ValueError("JSON must be a list or contain 'incidents'")
            rows = data
        elif p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")

        self.original_rows = [dict(r) for r in rows]
        self.rows = self._normalize(rows)
        return self.rows

    def _normalize(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_rows: List[Dict[str, Any]] = []
        for r in rows:
            nr: Dict[str, Any] = {}
            # Map columns
            for canon, candidates in CANONICAL_COLS.items():
                source_val: Any = None
                for name in candidates:
                    if name in r and r[name] not in (None, ""):
                        source_val = r[name]
                        break
                nr[canon] = source_val

            # Parse times
            start_dt = _parse_datetime(nr.get("start_time"))
            ack_dt = _parse_datetime(nr.get("ack_time"))
            end_dt = _parse_datetime(nr.get("end_time"))

            nr["start_time"] = start_dt
            nr["ack_time"] = ack_dt
            nr["end_time"] = end_dt

            # Normalize severity and strings
            nr["severity"] = _normalize_severity(nr.get("severity"))
            for k in ["id", "title", "description", "service", "status", "cause"]:
                if nr.get(k) is None:
                    nr[k] = ""
                else:
                    nr[k] = str(nr[k]).strip()

            # Tags
            tags = nr.get("tags")
            if isinstance(tags, list):
                nr["tags"] = [str(t) for t in tags]
            elif isinstance(tags, str):
                # Try common separators
                if ";" in tags:
                    nr["tags"] = [t.strip() for t in tags.split(";") if t.strip()]
                elif "," in tags:
                    nr["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
                else:
                    nr["tags"] = [tags.strip()] if tags.strip() else []
            else:
                nr["tags"] = []

            # Impact minutes
            try:
                nr["impact_minutes"] = float(nr["impact_minutes"]) if nr.get("impact_minutes") not in (None, "") else math.nan
            except Exception:
                nr["impact_minutes"] = math.nan

            # Derived durations
            ttr = (end_dt - start_dt).total_seconds() / 60.0 if start_dt and end_dt else math.nan
            tta = (ack_dt - start_dt).total_seconds() / 60.0 if start_dt and ack_dt else math.nan
            nr["ttr_minutes"] = ttr if ttr >= 0 else math.nan
            nr["tta_minutes"] = tta if tta >= 0 else math.nan

            # Week label (ISO week)
            if start_dt:
                iso = start_dt.isocalendar()  # (year, week, weekday)
                nr["week"] = f"{iso.year}-W{iso.week:02d}"
            else:
                nr["week"] = "unknown"

            normalized_rows.append(nr)
        return normalized_rows

    def compute_metrics(self, rows: Optional[List[Dict[str, Any]]] = None) -> Metrics:
        if rows is None:
            rows = self.rows
        total = len(rows)

        def collect(field: str) -> List[float]:
            vals: List[float] = []
            for r in rows:
                v = r.get(field)
                if isinstance(v, (int, float)) and not math.isnan(v):
                    vals.append(float(v))
            return vals

        ttr_vals = collect("ttr_minutes")
        tta_vals = collect("tta_minutes")

        mttr = sum(ttr_vals) / len(ttr_vals) if ttr_vals else None
        mtta = sum(tta_vals) / len(tta_vals) if tta_vals else None

        p50_mttr = _percentile(ttr_vals, 0.5) if ttr_vals else None
        p90_mttr = _percentile(ttr_vals, 0.9) if ttr_vals else None
        p50_mtta = _percentile(tta_vals, 0.5) if tta_vals else None
        p90_mtta = _percentile(tta_vals, 0.9) if tta_vals else None

        sev_counts: Dict[str, int] = {}
        svc_counts: Dict[str, int] = {}
        weekly_counts: Dict[str, int] = {}
        for r in rows:
            sev = r.get("severity") or "unknown"
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
            svc = (r.get("service") or "").strip().lower() or "unknown"
            svc_counts[svc] = svc_counts.get(svc, 0) + 1
            wk = r.get("week") or "unknown"
            weekly_counts[wk] = weekly_counts.get(wk, 0) + 1

        weekly_list = sorted(weekly_counts.items(), key=lambda x: x[0])

        return Metrics(
            total_incidents=total,
            mttr_minutes=mttr,
            mtta_minutes=mtta,
            incidents_by_severity=sev_counts,
            incidents_by_service=svc_counts,
            weekly_count=weekly_list,
            p50_mttr=p50_mttr,
            p90_mttr=p90_mttr,
            p50_mtta=p50_mtta,
            p90_mtta=p90_mtta,
        )

    def find_similar_clusters(self, rows: Optional[List[Dict[str, Any]]] = None, threshold: float = 0.6) -> List[List[int]]:
        if rows is None:
            rows = self.rows
        tokens = [
            _tokenize(f"{r.get('title','')} {r.get('description','')}") for r in rows
        ]
        n = len(tokens)
        visited = set()
        clusters: List[List[int]] = []
        for i in range(n):
            if i in visited:
                continue
            group = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j in visited:
                    continue
                sim = _jaccard(tokens[i], tokens[j])
                if sim >= threshold:
                    group.append(j)
                    visited.add(j)
            if len(group) > 1:
                clusters.append(group)
        return clusters

    def top_keywords(self, rows: Optional[List[Dict[str, Any]]] = None, top_n: int = 20) -> List[Tuple[str, int]]:
        if rows is None:
            rows = self.rows
        counter: Dict[str, int] = {}
        for r in rows:
            text = f"{r.get('title','')} {r.get('description','')}"
            for tok in _tokenize(text):
                if len(tok) <= 2:
                    continue
                counter[tok] = counter.get(tok, 0) + 1
        items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return items[:top_n]

    def suggested_root_causes(self, rows: Optional[List[Dict[str, Any]]] = None) -> Dict[int, str]:
        if rows is None:
            rows = self.rows
        keywords_to_cause = [
            (re.compile(r"(deploy|release|rollout|migration)"), "change related"),
            (re.compile(r"(cpu|memory|oom|disk|capacity|utilization)"), "resource saturation"),
            (re.compile(r"(network|dns|latency|timeout|connect|ssl)"), "networking"),
            (re.compile(r"(db|database|query|replica|index)"), "database"),
            (re.compile(r"(cache|redis|memcached)"), "caching"),
            (re.compile(r"(feature flag|flag|toggle)"), "feature flag"),
            (re.compile(r"(config|configuration)"), "configuration"),
            (re.compile(r"(ddos|attack|security|xss|csrf|rce)"), "security"),
            (re.compile(r"(dependency|third[- ]party|vendor)"), "third-party dependency"),
        ]
        suggestions: Dict[int, str] = {}
        for idx, r in enumerate(rows):
            text = f"{r.get('title','')} {r.get('description','')}".lower()
            for pattern, label in keywords_to_cause:
                if pattern.search(text):
                    suggestions[idx] = label
                    break
        return suggestions

    def to_dicts(self, rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        if rows is None:
            rows = self.rows

        def ts2iso(dt: Optional[datetime]) -> Optional[str]:
            if not isinstance(dt, datetime):
                return None
            return dt.astimezone(timezone.utc).isoformat()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": r.get("id") or None,
                "title": r.get("title") or None,
                "description": r.get("description") or None,
                "severity": r.get("severity") or None,
                "service": r.get("service") or None,
                "status": r.get("status") or None,
                "tags": r.get("tags") or [],
                "cause": r.get("cause") or None,
                "start_time": ts2iso(r.get("start_time")),
                "ack_time": ts2iso(r.get("ack_time")),
                "end_time": ts2iso(r.get("end_time")),
                "impact_minutes": (float(r.get("impact_minutes")) if isinstance(r.get("impact_minutes"), (int, float)) and not math.isnan(r.get("impact_minutes")) else None),
                "ttr_minutes": (float(r.get("ttr_minutes")) if isinstance(r.get("ttr_minutes"), (int, float)) and not math.isnan(r.get("ttr_minutes")) else None),
                "tta_minutes": (float(r.get("tta_minutes")) if isinstance(r.get("tta_minutes"), (int, float)) and not math.isnan(r.get("tta_minutes")) else None),
                "week": r.get("week") or None,
            })
        return out