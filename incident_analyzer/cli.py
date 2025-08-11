from __future__ import annotations

import pathlib
from typing import Optional

import typer
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn

from .analyzer import IncidentAnalyzer
from .report import render_html

app = typer.Typer(add_completion=False, help="Incident Analyzer CLI")


@app.command()
def analyze(path: str = typer.Argument(..., help="Path to incidents file (.json or .csv)"),
            out: Optional[str] = typer.Option(None, "--out", help="Output HTML report path"),
            preview: int = typer.Option(20, "--preview", help="Number of incidents to preview")):
    analyzer = IncidentAnalyzer()
    analyzer.load(path)
    metrics = analyzer.compute_metrics()

    template_dir = str(pathlib.Path(__file__).parent / "templates")
    html = render_html(analyzer, metrics, template_dir=template_dir, incidents_preview=preview)

    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(html, encoding="utf-8")
        typer.echo(f"Report written to {out}")
    else:
        typer.echo(html)


@app.command()
def serve(port: int = typer.Option(8000, "--port", help="Port to listen on")):
    analyzer = IncidentAnalyzer()

    api = FastAPI(title="Incident Analyzer")

    @api.get("/")
    async def index():  # type: ignore[no-redef]
        return HTMLResponse("""
        <html>
        <head><title>Incident Analyzer</title></head>
        <body style='font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial'>
          <h2>Incident Analyzer</h2>
          <form method="post" action="/analyze" enctype="multipart/form-data">
            <input type="file" name="file" accept=".json,.csv" />
            <button type="submit">Analyze</button>
          </form>
          <p>Upload a JSON or CSV file of incidents to get an HTML report.</p>
        </body></html>
        """)

    @api.post("/analyze")
    async def analyze_upload(file: UploadFile = File(...)):  # type: ignore[no-redef]
        tmp_path = pathlib.Path("/tmp") / file.filename
        data = await file.read()
        tmp_path.write_bytes(data)
        analyzer.load(str(tmp_path))
        metrics = analyzer.compute_metrics()
        template_dir = str(pathlib.Path(__file__).parent / "templates")
        html = render_html(analyzer, metrics, template_dir=template_dir, incidents_preview=50)
        return HTMLResponse(html)

    uvicorn.run(api, host="0.0.0.0", port=port)


if __name__ == "__main__":
    app()