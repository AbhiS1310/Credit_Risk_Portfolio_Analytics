"""Streamlit entry point. Run from the repository root with streamlit run app/app.py."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from threadpoolctl import threadpool_limits

from src.analytics import apply_filters, grouped, kpis
from src.common import ARTIFACTS, PROCESSED
from src.modeling import account_drivers
from src.qa import LLMAdapter, QAError, execute_query
from src.reviews import ReviewRepository, export_reviews_csv, make_review_engine
from src.runtime import load_artifacts

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="LendScope | Portfolio intelligence", page_icon="◈", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif;}
.block-container {padding-top:2.4rem;max-width:1500px;}
h1 {letter-spacing:-1.4px;font-weight:700!important;}
h2,h3 {letter-spacing:-.5px;}
[data-testid="stMetric"] {background:white;border:1px solid #E0E7EF;border-radius:12px;padding:18px;}
[data-testid="stMetricLabel"] {color:#5B6B82;font-size:.8rem;}
[data-testid="stSidebar"] {border-right:1px solid #E0E7EF;}
.eyebrow {font-size:11px;font-weight:700;letter-spacing:2px;color:#087F8C;text-transform:uppercase;}
.subtitle {color:#607086;font-size:15px;margin-top:-8px;margin-bottom:24px;}
.brand {font-size:25px;font-weight:700;letter-spacing:-1px;color:#123454;margin-bottom:0;}
.pill {background:#E0F2EF;color:#087568;border-radius:20px;padding:5px 12px;font-size:12px;font-weight:600;}
</style>""", unsafe_allow_html=True)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#087F8C", "#355C96", "#D19A47", "#7C6EA4", "#DD7868"]


@st.cache_resource
def built_data(manifest_stamp):
    return load_artifacts()


@st.cache_data
def detail(name, build_id):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@st.cache_resource
def database_engine(backend, url, sqlite_path):
    return make_review_engine({"REVIEW_STORAGE": backend, "DATABASE_URL": url, "REVIEW_SQLITE_PATH": sqlite_path})


def settings():
    values = {k: os.getenv(k, "") for k in ["LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "DATABASE_URL", "REVIEW_STORAGE", "REVIEW_SQLITE_PATH"]}
    try:
        for key in values:
            if key in st.secrets:
                values[key] = str(st.secrets[key])
    except FileNotFoundError:
        pass
    values["LLM_PROVIDER"] = values["LLM_PROVIDER"] or "groq"
    values["REVIEW_STORAGE"] = values["REVIEW_STORAGE"] or "sqlite"
    return values


def chart(fig):
    fig.update_layout(margin=dict(l=8, r=8, t=35, b=8), font=dict(family="DM Sans, Arial", color="#334761"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig, use_container_width=True)


def guidance(kind):
    with st.expander("How this AI feature works · limitations and human review"):
        if kind == "qa":
            st.markdown("**Trigger:** submit a question. The LLM proposes an allowlisted query; Python calculates every number.\n\n**False positive:** a valid query represents the wrong interpretation. **False negative:** a supported question is refused.\n\n**Human review:** inspect the query interpretation, filters, counts, and source evidence before using the answer. Only schema, categorical filters, and your question reach the provider; no customer records or review notes are sent.")
        else:
            st.markdown("**Trigger:** each data build recomputes peer rules and Isolation Forest.\n\n**False positive:** legitimate behavior is unusual. **False negative:** meaningful deterioration remains below thresholds or looks typical among risky peers.\n\n**Human review:** inspect payment history and peer comparisons, then confirm or dismiss. A flag is not proof of default or misconduct. The forest percentile is unusualness, not default probability. Bureau reasons describe a customer snapshot and can repeat across their accounts.")


def overview(frame, data):
    st.markdown('<div class="eyebrow">Portfolio workspace</div>', unsafe_allow_html=True)
    st.title("A clearer view of portfolio risk.")
    st.markdown('<div class="subtitle">Find concentrations, understand behavior, and decide where to look next.</div>', unsafe_allow_html=True)
    summary = kpis(frame)
    columns = st.columns(5)
    for col, label, value in zip(columns, ["Loan accounts", "Unique customers", "Observed default rate", "Observed exposure", "Original principal"], [f"{summary['account_count']:,}", f"{summary['customer_count']:,}", f"{summary['default_rate']:.2%}", f"${summary['observed_exposure']/1e6:,.2f}m" if summary['observed_exposure'] is not None else "Unknown", f"${summary['original_principal']/1e6:,.2f}m" if summary['original_principal'] is not None else "Unknown"]):
        col.metric(label, value)
    st.caption(f"{summary['default_count']:,} defaults / {summary['labeled_count']:,} labeled accounts · Exposure coverage: {summary['exposure_coverage']:,} accounts. Latest observed balances span different reporting dates; USD throughout.")
    st.caption("Data-quality note: the synthetic sources include chronology inconsistencies (including collections before origination). See the quality report below; scores are retrospective, not forecasts.")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Where risk concentrates")
        dim = st.selectbox("Profile dimension", ["loan_type", "employment_type", "region", "income_band", "age_band"], format_func=lambda x: x.replace("_", " ").title())
        groups = grouped(frame, [dim]).sort_values("default_rate", ascending=True)
        fig = px.bar(groups, x="default_rate", y=dim, orientation="h", color="default_rate", color_continuous_scale=["#C9E9E4", "#087F8C"], hover_data=["account_count", "default_count", "observed_exposure"])
        fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%", xaxis_title="Observed account default rate", yaxis_title=None)
        chart(fig)
    with right:
        st.subheader("Exposure by product")
        products = grouped(frame, ["loan_type"])
        fig = px.pie(products, values="observed_exposure", names="loan_type", hole=0.68)
        fig.update_traces(textinfo="percent", textposition="outside")
        fig.update_layout(legend=dict(orientation="h", y=-0.05), showlegend=True)
        chart(fig)
    st.subheader("Employment × income")
    cross = grouped(frame, ["employment_type", "income_band"])
    pivot = cross.pivot(index="employment_type", columns="income_band", values="default_rate")
    pivot = pivot.reindex(columns=[c for c in ["<25k", "25k-<50k", "50k-<100k", "100k+", "Unknown"] if c in pivot])
    fig = px.imshow(pivot, text_auto=".1%", color_continuous_scale=["#EEF5F5", "#087F8C"], aspect="auto", labels={"color": "Default rate"})
    chart(fig)
    st.caption("Rates are descriptive. Inspect account counts below; cells with fewer than 30 accounts are small samples.")
    with st.expander("Cross-cut counts and exact figures"):
        cross["small_sample"] = cross.account_count < 30
        st.dataframe(cross, hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Origination cohorts")
        cohort = grouped(frame, ["origination_year"])
        fig = px.line(cohort, x="origination_year", y="default_rate", markers=True, hover_data=["account_count", "default_count"])
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Origination year", yaxis_title="Observed eventual label rate")
        chart(fig)
    with right:
        st.subheader("Repayment behavior")
        repayment = detail("repayment_history", data["manifest"]["build_id"])
        trend = repayment[repayment.account_id.isin(frame.account_id)].groupby("payment_month").agg(mean_dpd=("days_past_due", "mean"), accounts=("account_id", "nunique")).reset_index()
        chart(px.line(trend, x="payment_month", y="mean_dpd", markers=True, labels={"payment_month": "Month since origination", "mean_dpd": "Mean days past due"}, hover_data=["accounts"]))
    with st.expander("Five measured insights · full portfolio"):
        st.caption("These build-time insights describe the full portfolio, regardless of active filters. They are associations in synthetic data.")
        for insight in data["insights"]:
            value = f"{insight['value']:.2%}" if insight["metric"] == "default_rate" else f"${insight['value']:,.2f}"
            st.markdown(f"**{insight['title']}** — {', '.join(insight['segment'].values())}: **{value}** ({insight['defaults']:,} defaults / {insight['accounts']:,} accounts). {insight['so_what']}")
    st.subheader("Accounts to explore")
    st.caption("Select an account in Account Detail to inspect score, drivers, and source history. Scores are retrospective model outputs.")
    st.dataframe(frame[["account_id", "loan_type", "region", "employment_type", "latest_outstanding", "risk_score", "target_default", "split"]].sort_values("risk_score", ascending=False).head(250), hide_index=True, use_container_width=True)


def account_view(frame, data):
    st.markdown('<div class="eyebrow">Account investigation</div>', unsafe_allow_html=True)
    st.title("Understand the account behind the score.")
    options = frame.account_id.tolist()
    if st.session_state.get("account_choice") not in options:
        st.session_state["account_choice"] = options[0]
    account = st.selectbox("Account", options, key="account_choice")
    row = frame[frame.account_id == account]
    record = row.iloc[0]
    st.caption(f"{record.customer_id} · {record.loan_type} · {record.region} · {record.employment_type}")
    a, b, c, d = st.columns(4)
    a.metric("Retrospective model score", f"{record.risk_score:.1%}")
    b.metric("Observed default label", "Defaulted" if record.target_default == 1 else "Performing")
    c.metric("Latest observed balance", f"${record.latest_outstanding:,.2f}")
    d.metric("Evaluation partition", str(record['split']).title())
    st.info("This is retrospective classification, not a validated future probability of default. Label dates and a prediction horizon are unavailable. Train-partition scores are in-sample.")
    with threadpool_limits(limits=2):
        drivers, explanation = account_drivers(data["model"], row)
    left, right = st.columns(2)
    with left:
        st.subheader("Key model drivers")
        fig = px.bar(drivers.sort_values("contribution"), x="contribution", y="feature", orientation="h", color="contribution", color_continuous_scale=["#087F8C", "#DD7868"])
        fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
        chart(fig); st.caption(explanation)
    repayment = detail("repayment_history", data["manifest"]["build_id"])
    history = repayment[repayment.account_id == account].sort_values("payment_month")
    with right:
        st.subheader("Repayment history")
        chart(px.line(history, x="payment_month", y="days_past_due", markers=True, labels={"payment_month": "Month since origination", "days_past_due": "Days past due"}))
    with st.expander("Source repayment records", expanded=False):
        st.dataframe(history, hide_index=True, use_container_width=True)
    st.subheader("Bureau snapshot · customer level")
    st.caption("Undated reporting snapshot. These lines can postdate origination and repeat across this customer's accounts.")
    bureau = detail("bureau_credit", data["manifest"]["build_id"])
    st.dataframe(bureau[bureau.customer_id == record.customer_id], hide_index=True, use_container_width=True)
    st.subheader("Collection activity · excluded from modeling")
    events = detail("collections_events", data["manifest"]["build_id"])
    st.dataframe(events[events.account_id == account].sort_values("event_date"), hide_index=True, use_container_width=True)


def qa_view(all_accounts, filters, data, config):
    st.markdown('<div class="eyebrow">Ask the portfolio</div>', unsafe_allow_html=True)
    st.title("Ask a question. Inspect the evidence.")
    st.caption("Try: Which loan type has the highest default rate? · What is the observed exposure by region? · Do not enter personal information.")
    adapter = LLMAdapter(config)
    if not adapter.configured:
        st.warning("LLM Q&A is not configured. Set LLM_PROVIDER=groq, LLM_MODEL, and LLM_API_KEY in .env or Streamlit secrets. Analytics and anomaly detection remain available.")
    else:
        st.caption(f"Provider: {adapter.provider} · Model: {adapter.model} · Structured planning; local calculations")
    guidance("qa")
    with st.form("portfolio_question"):
        question = st.text_area("Your question", max_chars=1000, placeholder="Which loan type has the highest default rate?")
        submitted = st.form_submit_button("Ask portfolio", disabled=not adapter.configured or st.session_state.get("qa_busy", False), type="primary")
    fingerprint = json.dumps(filters, sort_keys=True) + data["manifest"]["build_id"]
    if submitted:
        st.session_state["qa_busy"] = True
        st.session_state.pop("qa_answer", None)
        try:
            with st.spinner("Planning a constrained query and checking the data…"):
                plan = adapter.plan_query(question, all_accounts, filters)
                answer = execute_query(all_accounts, plan, filters, data["manifest"]["build_id"])
            st.session_state["qa_answer"] = (fingerprint, answer)
        except (QAError, ValueError) as exc:
            st.error(str(exc) if isinstance(exc, QAError) else "Query validation failed. No answer was generated.")
        finally:
            st.session_state["qa_busy"] = False
    saved = st.session_state.get("qa_answer")
    if saved and saved[0] == fingerprint:
        answer = saved[1]
        st.markdown(answer["answer"])
        if answer.get("records"):
            st.dataframe(pd.DataFrame(answer["records"]), hide_index=True, use_container_width=True)
        with st.expander("Interpretation, exact figures, and source evidence", expanded=True):
            st.json(answer)


def review_view(frame, data, config):
    st.markdown('<div class="eyebrow">Human review queue</div>', unsafe_allow_html=True)
    st.title("Unusual patterns, ready for a closer look.")
    guidance("anomaly")
    scope_ids = set(frame.account_id)
    flags = [r for r in data["anomalies"] if r["account_id"] in scope_ids]
    if not flags:
        st.info("No flagged accounts in the active scope."); return
    flags.sort(key=lambda r: (-r["anomaly_percentile"], r["account_id"]))
    valid = {(r["account_id"], r["anomaly_version"]) for r in data["anomalies"]}
    repository, history = None, []
    backend = config["REVIEW_STORAGE"]
    if backend == "sqlite":
        st.warning("SQLite assessment mode: decisions are saved to this app's local database. Streamlit Community Cloud does not guarantee local-file persistence; reviews may be lost on restart, redeploy, or storage reset. Download the CSV to retain a copy.")
    else:
        st.caption("Managed Postgres mode · review history is stored outside the app instance.")
    try:
        repository = ReviewRepository(database_engine(backend, config.get("DATABASE_URL", ""), config.get("REVIEW_SQLITE_PATH", "")), valid)
        history = repository.history()
    except (SQLAlchemyError, ValueError, OSError):
        repository = None
        st.error("Review storage unavailable. Decisions cannot be saved. Check SQLite file permissions, or the selected Postgres configuration and initialized schema.")
    export_history = [event for event in history if event["account_id"] in scope_ids]
    st.download_button("Download review history (CSV)", export_reviews_csv(export_history), file_name="lendscope_review_history.csv", mime="text/csv", disabled=repository is None, key="review_export")
    st.caption(f"CSV contains {len(export_history):,} committed events across all detector versions for accounts in the active portfolio scope. Review-status filtering does not change the export. Timestamps are UTC.")
    latest = {}
    for event in history:
        latest.setdefault((event["account_id"], event["anomaly_version"]), event["decision"])
    queue = pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} | {"decision": latest.get((r["account_id"], r["anomaly_version"]), "pending")} for r in flags])
    a, b, c = st.columns(3)
    a.metric("Flagged accounts in scope", f"{len(flags):,}")
    b.metric("Pending review", f"{(queue.decision == 'pending').sum():,}")
    c.metric("Confirmed / dismissed", f"{(queue.decision == 'confirm').sum():,} / {(queue.decision == 'dismiss').sum():,}")
    st.caption(f"Detector version {flags[0]['anomaly_version']} · Rankings are unusualness percentiles, not probabilities.")
    status = st.selectbox("Review status", ["All", "pending", "confirm", "dismiss"])
    visible = queue if status == "All" else queue[queue.decision == status]
    st.dataframe(visible.head(250), hide_index=True, use_container_width=True)
    if visible.empty:
        st.info("No accounts match this review status."); return
    account = st.selectbox("Flagged account to review", visible.account_id.tolist())
    flag = next(r for r in flags if r["account_id"] == account)
    st.subheader(f"Evidence for {account}")
    st.caption(f"Peer segment: {flag['peer_segment']} · {flag['peer_count']:,} other accounts · Feature coverage {flag['feature_coverage']}/{flag['feature_total']} · Rule flag: {flag['rule_flag']} · Forest flag: {flag['forest_flag']}")
    st.dataframe(pd.DataFrame(flag["reasons"]), hide_index=True, use_container_width=True)
    repayment = detail("repayment_history", data["manifest"]["build_id"])
    record = frame[frame.account_id == account].iloc[0]
    left, right = st.columns(2)
    with left:
        chart(px.line(repayment[repayment.account_id == account], x="payment_month", y="days_past_due", markers=True, title="DPD by month since origination"))
    with right:
        st.write("Account behavioral features")
        st.dataframe(frame.loc[frame.account_id == account, ["repay_payment_ratio", "recent_dpd", "previous_dpd", "dpd_deterioration", "recent_missed", "previous_missed", "bureau_utilization"]].T.rename(columns={record.name: "Observed"}), use_container_width=True)
    token_key = f"review_token:{account}:{flag['anomaly_version']}"
    st.session_state.setdefault(token_key, str(uuid4()))
    with st.form("review_decision"):
        decision = st.radio("Decision", ["confirm", "dismiss"], horizontal=True)
        note = st.text_area("Review note (optional)", max_chars=1000)
        submitted = st.form_submit_button("Save decision", disabled=repository is None, type="primary")
    if submitted and repository:
        try:
            repository.append(st.session_state[token_key], account, flag["anomaly_version"], decision, note)
            st.session_state[token_key] = str(uuid4())
            st.session_state["review_success"] = f"Saved {decision} for {account}."
            st.rerun()
        except (SQLAlchemyError, ValueError):
            st.error("Decision was not confirmed saved. Check database availability and retry; the submission ID prevents duplicate writes.")
    if st.session_state.get("review_success"):
        st.success(st.session_state.pop("review_success"))
    account_events = [event for event in history if event["account_id"] == account]
    if account_events:
        with st.expander("Committed review history · prior versions stay separate"):
            st.dataframe(pd.DataFrame(account_events), hide_index=True, use_container_width=True)


def diagnostics(data):
    with st.expander("Model evaluation, data quality, and reproducibility"):
        st.warning(data["metrics"]["limitation"])
        rows = []
        for name in ["logistic", "histogram"]:
            for partition in ["validation", "test"]:
                rows.append({"model": name, "partition": partition, **data["metrics"][name][partition]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Selected: {data['metrics']['selected']} · Selection uses validation AUC, with ties within 0.005 favoring logistic regression. Test results do not select the model.")
        st.dataframe(data["importance"].head(15), hide_index=True, use_container_width=True)
        st.caption("Validation permutation importance: mean change in ROC AUC over five repeats.")
        st.json(data["quality"])
        st.json(data["manifest"])
        for name, label in [("architecture.pdf", "Architecture PDF"), ("solution_overview.pdf", "Solution overview PDF")]:
            path = ROOT / "docs" / name
            if path.exists():
                st.download_button(label, path.read_bytes(), file_name=name, mime="application/pdf", key=name)


def main():
    try:
        stamp = (ARTIFACTS / "manifest.json").stat().st_mtime_ns
        data = built_data(stamp)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        st.error(f"Application data cannot load: {exc}")
        st.stop()
    config = settings()
    accounts = data["accounts"]
    with st.sidebar:
        st.markdown('<div class="brand">◈ LendScope</div>', unsafe_allow_html=True)
        st.caption("PORTFOLIO INTELLIGENCE")
        st.divider()
        view = st.radio("Workspace", ["Portfolio", "Account Detail", "Ask the Portfolio", "Anomaly Review"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Portfolio scope**")
        filters = {}
        for field, label in [("region", "Region"), ("loan_type", "Loan type"), ("employment_type", "Employment")]:
            choices = sorted(accounts[field].dropna().unique().tolist())
            selected = st.multiselect(label, choices, key=f"filter_{field}", placeholder="All")
            if selected:
                filters[field] = selected
        if st.button("Clear filters", use_container_width=True):
            for field in ["region", "loan_type", "employment_type"]:
                st.session_state.pop(f"filter_{field}", None)
            st.rerun()
        st.divider()
        st.caption("SYNTHETIC DATA · USD")
        st.caption("Retrospective scores. Human judgment required.")
        st.caption(f"Build {data['manifest']['build_id'][:10]}")
    frame = apply_filters(accounts, filters)
    st.caption("SCOPE · " + (" · ".join(f"{k.replace('_', ' ')}: {', '.join(v)}" for k, v in filters.items()) if filters else "Full portfolio") + f" · {len(frame):,} accounts")
    if frame.empty:
        st.info("No accounts match these filters. Adjust or clear the portfolio scope.")
        return
    if view == "Portfolio":
        overview(frame, data)
    elif view == "Account Detail":
        account_view(frame, data)
    elif view == "Ask the Portfolio":
        qa_view(accounts, filters, data, config)
    else:
        review_view(frame, data, config)
    diagnostics(data)
    st.divider()
    st.caption("LendScope · Synthetic credit-risk assessment · Decision support, with verifiable evidence.")


if __name__ == "__main__":
    main()
