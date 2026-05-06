import os
from datetime import datetime, timezone

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics.charts.lineplots import LinePlot
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

REPORTS_DIR = "static/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SEVERITY_COLORS = {
    "HIGH":   "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW":    "#3b82f6",
}


def generate_pdf(data: dict) -> str:
    out_path = os.path.join(REPORTS_DIR, f"report_{data['exam_id'][:8]}.pdf")

    if not REPORTLAB_AVAILABLE:
        _write_text_report(data, out_path.replace(".pdf", ".txt"))
        return out_path.replace(".pdf", ".txt")

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story  = []

    # ── Header ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )
    story.append(Paragraph("🔒 ProctorAI — Exam Integrity Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#6366f1")))
    story.append(Spacer(1, 0.4*cm))

    # ── Summary cards ──────────────────────────────────────────────────
    sub  = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
    bold = ParagraphStyle("Bold", parent=styles["Normal"], fontSize=11, fontName="Helvetica-Bold")

    summary_data = [
        ["Candidate",     data["candidate_name"]],
        ["Exam",          data.get("exam_title", "—")],
        ["Status",        data["status"]],
        ["Start",         data["start_time"][:19].replace("T", "  ")],
        ["End",           (data.get("end_time") or "—")[:19].replace("T", "  ")],
        ["Duration",      f"{data['duration_minutes']:.1f} minutes"],
        ["Total Violations", str(data["total_violations"])],
        ["Highest Risk Score", f"{data['highest_risk_score']} / 100"],
    ]
    t = Table(summary_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0,0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── Violations by type ─────────────────────────────────────────────
    story.append(Paragraph("Violations by Type", styles["Heading2"]))
    vbt = data.get("violations_by_type", {})
    if vbt:
        rows = [["Type", "Count"]] + [[k, str(v)] for k, v in vbt.items()]
        vt = Table(rows, colWidths=[8*cm, 4*cm])
        vt.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#6366f1")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("ALIGN",       (1, 0), (1, -1), "CENTER"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ]))
        story.append(vt)
    else:
        story.append(Paragraph("No violations recorded.", styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Violation timeline ──────────────────────────────────────────────
    story.append(Paragraph("Violation Timeline", styles["Heading2"]))
    violations = data.get("violations", [])
    if violations:
        rows = [["#", "Time", "Type", "Severity", "Risk Score"]]
        for i, v in enumerate(violations[:50], 1):
            ts = v["timestamp"][:19].replace("T", " ")
            rows.append([str(i), ts, v["type"], v["severity"], str(v["risk_score"])])
        vl = Table(rows, colWidths=[1*cm, 4.5*cm, 5*cm, 2.5*cm, 2.5*cm])
        sev_colors_rl = {
            "HIGH":   colors.HexColor("#fee2e2"),
            "MEDIUM": colors.HexColor("#fef3c7"),
            "LOW":    colors.HexColor("#dbeafe"),
        }
        style = [
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ]
        for idx, v in enumerate(violations[:50], 1):
            bg = sev_colors_rl.get(v["severity"], colors.white)
            style.append(("BACKGROUND", (3, idx), (3, idx), bg))
        vl.setStyle(TableStyle(style))
        story.append(vl)
    else:
        story.append(Paragraph("No violations recorded — clean exam! ✓", styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Snapshots ──────────────────────────────────────────────────────
    snap_violations = [v for v in violations if v.get("snapshot_path")]
    if snap_violations:
        story.append(Paragraph("Violation Snapshots", styles["Heading2"]))
        for v in snap_violations[:6]:
            snap_path = v["snapshot_path"]
            if isinstance(snap_path, str) and snap_path.startswith("/static/"):
                snap_path = snap_path.lstrip("/")
            if os.path.exists(snap_path):
                try:
                    img = RLImage(snap_path, width=6*cm, height=4.5*cm)
                    story.append(img)
                    story.append(Paragraph(
                        f"{v['type']} — {v['timestamp'][:19]} — Score: {v['risk_score']}",
                        sub
                    ))
                    story.append(Spacer(1, 0.3*cm))
                except Exception:
                    continue

    # ── Footer ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        f"Generated by ProctorAI · HACKHIVE-2k26 · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey, alignment=1),
    ))

    doc.build(story)
    return out_path


def _write_text_report(data: dict, path: str):
    with open(path, "w") as f:
        f.write("ProctorAI — Exam Integrity Report\n")
        f.write("=" * 50 + "\n")
        for k, v in data.items():
            if k != "violations":
                f.write(f"{k}: {v}\n")
        f.write("\nViolations:\n")
        for v in data.get("violations", []):
            f.write(f"  {v['timestamp']} | {v['type']} | {v['severity']} | score={v['risk_score']}\n")
