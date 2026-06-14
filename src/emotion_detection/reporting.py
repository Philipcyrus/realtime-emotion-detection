"""Local JSON and HTML report export for emotion detection sessions."""

from __future__ import annotations

import html
import json
from pathlib import Path

from emotion_detection.session import SessionSummary


def export_session_report(summary: SessionSummary, output_dir: Path) -> dict[str, Path]:
    """Write privacy-safe JSON and HTML reports for a session."""

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(summary.session_name)
    json_path = output_dir / f"{slug}.json"
    html_path = output_dir / f"{slug}.html"

    json_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html_report(summary), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def render_html_report(summary: SessionSummary) -> str:
    distribution_rows = "\n".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in summary.emotion_distribution.items()
    ) or "<tr><td colspan=\"2\">No emotions detected</td></tr>"

    timeline_rows = "\n".join(
        "<tr>"
        f"<td>{event['offset_seconds']}</td>"
        f"<td>{html.escape(str(event['label']))}</td>"
        f"<td>{event['confidence']}</td>"
        f"<td>{html.escape(str(event['source']))}</td>"
        "</tr>"
        for event in summary.timeline[:200]
    ) or "<tr><td colspan=\"4\">No face predictions recorded</td></tr>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Emotion Detection Report - {html.escape(summary.session_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; }}
    header {{ border-bottom: 2px solid #1f7a8c; margin-bottom: 24px; }}
    h1 {{ margin-bottom: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #ccd6dd; border-radius: 8px; padding: 12px; background: #f8fbfc; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d7dee4; padding: 8px; text-align: left; }}
    th {{ background: #eaf3f6; }}
    section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <header>
    <h1>Emotion Detection Report</h1>
    <p>{html.escape(summary.session_name)}</p>
  </header>
  <section class="grid">
    <div class="metric">Frames<strong>{summary.frames_processed}</strong></div>
    <div class="metric">Faces<strong>{summary.faces_detected}</strong></div>
    <div class="metric">Average FPS<strong>{summary.average_fps}</strong></div>
    <div class="metric">Average Confidence<strong>{summary.average_confidence}</strong></div>
  </section>
  <section>
    <h2>Emotion Distribution</h2>
    <table>
      <thead><tr><th>Emotion</th><th>Count</th></tr></thead>
      <tbody>{distribution_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Timeline</h2>
    <table>
      <thead><tr><th>Time</th><th>Emotion</th><th>Confidence</th><th>Source</th></tr></thead>
      <tbody>{timeline_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part) or "session"
