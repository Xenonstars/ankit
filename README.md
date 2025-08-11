Incident Analyzer

Quick start:

1) Create venv and install deps:
   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

2) Analyze sample incidents and generate HTML report:
   python -m incident_analyzer.cli analyze sample/incidents.json --out sample/report.html

3) Start web UI (upload JSON/CSV):
   python -m incident_analyzer.cli serve --port 8000