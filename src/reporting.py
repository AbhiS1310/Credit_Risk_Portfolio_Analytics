"""Create exactly two one-page submission PDFs using measured build outputs."""
from __future__ import annotations

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

from src.common import ROOT
from src.runtime import load_artifacts

NAVY = colors.HexColor("#172B4D")
TEAL = colors.HexColor("#087F8C")
MUTED = colors.HexColor("#52657C")
LIGHT = colors.HexColor("#EDF4F7")
WIDTH, HEIGHT = A4
MARGIN = 38
BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.1, leading=12.6, textColor=NAVY, spaceAfter=5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.2, leading=11)


def paragraph(canvas, text, x, y, width, style=BODY):
    p = Paragraph(text, style)
    _, height = p.wrap(width, 1000)
    p.drawOn(canvas, x, y-height)
    return y-height-7


def section(canvas, title, y):
    canvas.setFillColor(TEAL)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, y, title.upper())
    return y-14


def page(path, title, subtitle, build_id):
    c = Canvas(str(path), pagesize=A4, pageCompression=1)
    c.setTitle(title)
    c.setAuthor("Credit Risk Portfolio Assessment")
    c.setFillColor(NAVY); c.rect(0, HEIGHT-105, WIDTH, 105, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#6BD0CA")); c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, HEIGHT-30, "LENDSCOPE / CREDIT RISK PORTFOLIO")
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 23)
    c.drawString(MARGIN, HEIGHT-61, title)
    c.setFont("Helvetica", 9); c.drawString(MARGIN, HEIGHT-82, subtitle)
    c.setStrokeColor(colors.HexColor("#D9E4EC")); c.line(MARGIN, 31, WIDTH-MARGIN, 31)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.5)
    c.drawString(MARGIN, 19, f"Synthetic assessment | Build {build_id} | Retrospective analysis")
    c.drawRightString(WIDTH-MARGIN, 19, "1 / 1")
    return c


def box(c, title, text, x, top, width, height):
    c.setFillColor(LIGHT); c.roundRect(x, top-height, width, height, 8, fill=1, stroke=0)
    c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 10)
    c.drawString(x+12, top-18, title)
    return paragraph(c, text, x+12, top-27, width-24, SMALL)


def arrow(c, x, top, bottom):
    c.setStrokeColor(TEAL); c.setLineWidth(1.2); c.line(x, top, x, bottom+4)
    c.setFillColor(TEAL)
    p = c.beginPath(); p.moveTo(x-3, bottom+6); p.lineTo(x, bottom); p.lineTo(x+3, bottom+6); p.close()
    c.drawPath(p, fill=1, stroke=0)


def architecture(data):
    manifest = data["manifest"]
    c = page(ROOT / "docs" / "architecture.pdf", "Architecture", "Raw evidence to analyst decisions, with explicit AI boundaries.", manifest["build_id"])
    full = WIDTH-2*MARGIN
    top = HEIGHT-124
    box(c, "01  RAW SOURCES", "customers (10,000) + loans (14,685) + bureau (22,979) + repayments (176,220) + collections (8,635). Immutable CSVs, SHA-256 checksums.", MARGIN, top, full, 65)
    arrow(c, WIDTH/2, top-65, top-82)
    top -= 84
    box(c, "02  INGESTION & ACCOUNT FEATURES", "pandas / PyArrow: typed validation, quality report, category normalization. Bureau aggregates by customer; repayment and collections by account. Left joins preserve 14,685 loans. Detailed sources retained for drill-down.", MARGIN, top, full, 77)
    arrow(c, WIDTH/2, top-77, top-94)
    top -= 96
    gap = 12; w = (full-2*gap)/3
    box(c, "ANALYTICS", "pandas: counts, default rates, observed exposure, cohorts and employment x income. Every rate retains its denominator.", MARGIN, top, w, 106)
    box(c, "DEFAULT MODEL", "scikit-learn: logistic + histogram boosting. Customer-disjoint 60/20/20 split. Train-only preprocessing; AUC, KS, Gini; feature importance.", MARGIN+w+gap, top, w, 106)
    box(c, "ANOMALIES", "scikit-learn: peer rules + Isolation Forest. Unlabeled behavior, leave-one-out comparisons, deterministic review reasons.", MARGIN+2*(w+gap), top, w, 106)
    arrow(c, WIDTH/2, top-106, top-123)
    top -= 125
    box(c, "03  STREAMLIT DASHBOARD / PLOTLY", "Portfolio filters and risk/exposure charts | Account score, drivers and source histories | Ask the Portfolio | Anomaly review. Versioned parquet, joblib and JSON artifacts are verified before loading; startup never retrains.", MARGIN, top, full, 68)
    arrow(c, MARGIN+full/4, top-68, top-84); arrow(c, MARGIN+full*3/4, top-68, top-84)
    top -= 87
    half = (full-gap)/2
    box(c, "SERVER-SIDE LLM / GROQ / GPT-OSS", "HTTPS structured query planning. Send schema/metric definitions, categorical filter scope and analyst question only. Pydantic allowlists the result; pandas executes it; deterministic text cites exact local figures. No raw customer records or review notes leave the app.", MARGIN, top, half, 120)
    box(c, "SQLITE + CSV EXPORT", "SQLAlchemy + sqlite3. Append-only, versioned confirm/dismiss events with UUID, note and UTC time. CSV download retains a review copy. Streamlit local files may reset; no durable cloud guarantee. Optional managed Postgres backend.", MARGIN+half+gap, top, half, 120)
    y = top-137
    y = section(c, "Hosting, secrets & limitations", y)
    paragraph(c, "Python 3.11 / Streamlit Community Cloud target. The LLM key stays server-side; SQLite needs no database credentials. Export review CSVs before a cloud restart/redeploy. Managed Postgres is optional for durable storage. Missing default dates, undated bureau snapshots and inconsistent collection chronology prevent prospective validation.", MARGIN, y, full, SMALL)
    c.save()


def overview(data):
    full = WIDTH-2*MARGIN
    c = page(ROOT / "docs" / "solution_overview.pdf", "Solution overview", "An analyst workspace for portfolio monitoring and explainable investigation.", data["manifest"]["build_id"])
    y = HEIGHT-124
    y = section(c, "Analyst workflow", y)
    y = paragraph(c, "<b>Portfolio:</b> filter region, product and employment; inspect income/age cuts, observed exposure, cohorts and employment x income. <b>Account:</b> inspect retrospective score, signed drivers, repayment history, undated customer bureau lines and collections. <b>Ask:</b> query exact aggregates and inspect evidence. <b>Review:</b> investigate flags and record confirm/dismiss decisions in SQLite and export CSV.", MARGIN, y, full)
    y -= 2; y = section(c, "Measured model results / held-out test", y)
    test = data["metrics"][data["metrics"]["selected"]]["test"]
    hist = data["metrics"]["histogram"]["test"]
    y = paragraph(c, f"<b>Selected logistic regression:</b> AUC <b>{test['auc']:.4f}</b> | KS <b>{test['ks']:.4f}</b> | Gini <b>{test['gini']:.4f}</b>. Histogram boosting: AUC {hist['auc']:.4f}, KS {hist['ks']:.4f}, Gini {hist['gini']:.4f}. Test: {test['accounts']:,} loans / {test['defaults']:,} defaults. Customers split 60/20/20; train-only imputation/encoding; validation AUC selects the model. Bureau + repayment features; global permutation importance and signed logistic drivers. IDs, gender and collections excluded.", MARGIN, y, full)
    y = paragraph(c, "<b>Interpretation:</b> retrospective classification only, not a validated future PD. The target has no default date or horizon; bureau snapshots lack reporting dates. Monitor input/score drift and delayed-label AUC/KS/Gini/calibration after a prospective design is established.", MARGIN, y, full, SMALL)
    y -= 2; y = section(c, "Top three portfolio insights / full synthetic sample", y)
    for index in [0, 1, 4]:
        insight = data["insights"][index]
        segment = " / ".join(insight["segment"].values())
        y = paragraph(c, f"<b>{escape(segment)}:</b> observed default rate <b>{insight['value']:.2%}</b> ({insight['defaults']:,}/{insight['accounts']:,} accounts). {escape(insight['so_what'])}", MARGIN, y, full)
    y -= 2; y = section(c, "Two embedded AI features / human judgment required", y)
    y = paragraph(c, "<b>Ask the Portfolio - trigger:</b> analyst submits a question. Groq proposes a constrained query; Python computes numbers and provenance. Unsupported questions are declined. <b>False positive:</b> wrong interpretation yields a valid query. <b>False negative:</b> supported query is refused. <b>Human step:</b> verify interpretation, filters and exact source figures. Only definitions, categorical scope and question reach the provider.", MARGIN, y, full)
    y = paragraph(c, f"<b>Anomalies - trigger:</b> each build runs peer rules and Isolation Forest; current queue has {len(data['anomalies']):,} accounts. Flags show payment, DPD, missed-payment or bureau evidence. <b>False positive:</b> unusual legitimate behavior. <b>False negative:</b> missed deterioration or risky peers masking a problem. <b>Human step:</b> inspect history, confirm/dismiss with a note; SQLite keeps the versioned decision; CSV export retains a copy. Unusualness is not default probability.", MARGIN, y, full)
    y -= 2; y = section(c, "Assumptions & production needs", y)
    y = paragraph(c, "Latest observed outstanding is exposure, not a common-date bank balance. Twelve repayment months per loan; synthetic chronology includes collection events before origination, retained as quality warnings. No causal claims. Anonymous public demo reviews need authenticated identity in production. IT/data teams must provide timestamped snapshots, cutoff/outcome definitions, approved vendors and access, monitored hosting, backups, audit controls and model-risk review. SQLite files may be lost on hosted restart/redeploy; export CSV. Managed Postgres is optional for durability. Hosted link, repository and video are separate submission steps.", MARGIN, y, full, SMALL)
    if y < 42:
        raise ValueError(f"Solution overview would overflow one page: bottom {y}")
    c.save()


def render_documents():
    data = load_artifacts()
    architecture(data)
    overview(data)
    print("Generated docs/architecture.pdf and docs/solution_overview.pdf (one page each).")


if __name__ == "__main__":
    render_documents()
