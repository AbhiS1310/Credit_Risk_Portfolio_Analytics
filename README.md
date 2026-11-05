# LendScope · Credit Risk Portfolio Analytics

A working Streamlit workspace for monitoring a synthetic loan portfolio: portfolio cuts and exposure, account-level retrospective scores and drivers, grounded LLM Q&A, and anomaly investigation with recorded human decisions and CSV export.

**Local implementation is complete. Groq is configured and a real grounded Q&A request has passed.** Review saving now works without database credentials using SQLite, with CSV export. Streamlit cloud storage may reset; optional Postgres provides durable hosted history. The LLM is not simulated when credentials are missing. The recorded demo is linked below; hosted deployment and repository publication status must be confirmed separately.

## Run locally

Tested with **Python 3.11.17**. Use Python 3.11 in a virtual environment. From the repository root, run these three commands:

```shell
pip install -r requirements.txt
python -m src.build
streamlit run app/app.py
```

The first build copies the supplied five CSVs into `data/raw/` without changing their bytes, validates them, generates clean data and models, and creates the two one-page PDFs. On this machine the build takes about 25 seconds. Generated artifacts are included so deployment startup does not retrain. Only load the repository's own trusted joblib files.

Windows users can use `.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe` if the virtual environment is not activated. A workspace-local Python installation was used for development; `.cache/` and `.venv/` are ignored. `requirements.txt` pins direct dependencies; `requirements.lock.txt` records the entire tested environment, with conditional Windows packages marked for portability.

## Groq and database configuration

Copy `.env.example` to a local `.env` file and fill in values, or enter the equivalent top-level keys in Streamlit Community Cloud's secrets settings. `.env` and `.streamlit/secrets.toml` are ignored by Git.

- `LLM_PROVIDER=groq` selects Groq, using `https://api.groq.com/openai/v1` by default.
- `LLM_MODEL` is the exact structured-output-capable model ID enabled in your Groq account. It is deliberately not guessed.
- `LLM_API_KEY` is the provider key, read only on the server.
- `LLM_BASE_URL` optionally overrides the HTTPS API base. The adapter supports OpenAI-compatible chat-completions endpoints; `LLM_PROVIDER=xai` also supports xAI Grok; `LLM_PROVIDER=openai` selects that default endpoint. Other providers require an explicit compatible base URL.
- `REVIEW_STORAGE=sqlite` is the default: no database service or password is needed. `REVIEW_SQLITE_PATH` optionally overrides `data/reviews.db`.
- `DATABASE_URL` is only needed with `REVIEW_STORAGE=postgres`. The optional Postgres backend uses psycopg over TLS.

SQLite initializes its table automatically when Anomaly Review opens. Its runtime database is excluded from Git. Review decisions survive browser refreshes and new sessions while the file exists. **Streamlit Community Cloud does not guarantee local-file persistence**: reviews may be lost on restart, redeployment or storage reset. Use **Download review history (CSV)** to retain committed decisions with UTC timestamps. The export follows active portfolio filters, includes prior detector versions, and is independent of the review-status filter. See [deployment guide](docs/deployment.md).

For optional managed Postgres, set `REVIEW_STORAGE=postgres` and `DATABASE_URL`, then initialize its schema with `python -m src.reviews --init` using a database-owner connection. Use an app role with schema usage and SELECT/INSERT privileges for normal runtime. No automatic migration from SQLite is performed.

Without LLM credentials, Q&A submission is disabled with setup guidance. Provider failures do not produce invented figures. Storage failures disable saving visibly; a decision is shown as saved only after commit. Public demo reviews are anonymous, versioned, append-only and idempotent. SQLite is an explicit assessment backend, not a claim of durable hosted storage.

## What the application shows

- **Portfolio:** account/customer counts, default rate with denominator, latest observed exposure, original principal, required income/employment/region/age/product cuts, employment × income, origination cohorts, repayment-month trends, and five measured insights. Region, product, and employment filters apply across views.
- **Account Detail:** a selected account's retrospective score, observed default label, evaluation partition, signed model drivers, repayment records, customer-level bureau snapshot, and collection events.
- **Ask the Portfolio:** Groq plans a typed, allowlisted aggregate query. Local pandas calculations and deterministic formatting produce the answer with exact counts, scope, metric definitions, source fields, and build ID. No model-generated SQL/code executes. Only schema descriptions, categorical scope, and the question are sent to the provider; row-level customer data and review notes stay local.
- **Anomaly Review:** leave-one-out peer rules plus Isolation Forest identify unusual behavior. Reasons include payment ratios, DPD deterioration, missed payments, and bureau utilization. Reviewers inspect history and record confirm/dismiss decisions in SQLite (or optional Postgres), and download committed history as CSV.

Both AI views explain their trigger, false positives, false negatives, and required human review. The expandable diagnostics panel contains both models' validation/test metrics, permutation importance, the quality report, build manifest, and downloadable PDFs.

## Measured results

The supplied data has **14,685 loans**, **10,000 customers**, and **2,364 defaults (16.10%)**. Original principal totals **$601,119,531.64**; latest observed outstanding totals **$512,637,943.19**. Exposure is not a common-date bank balance.

Customer-grouped splits contain 6,000/2,000/2,000 customers for train/validation/test. Preprocessing is fitted only on train. Logistic regression won validation AUC against four histogram gradient boosting configurations; test results did not select the model.

- **Logistic regression:** test AUC **0.6768**, KS **0.2607**, Gini **0.3536**.
- **Histogram gradient boosting:** test AUC **0.6613**, KS **0.2416**, Gini **0.3226**.
- Held-out test: **2,948 accounts / 472 defaults**. Anomaly queue: **1,685 accounts**. Flag counts are not measured accuracy.

Example measured insights: mortgages have a 24.65% observed default rate (353/1,432); unemployed customers' accounts have 34.96% (251/718); the unemployed / below-$25k joint segment has 44.62% (87/195). These synthetic associations motivate investigation, not causal claims or automatic credit-policy changes. All five insights and implications are in `artifacts/insights.json` and the dashboard.

## Important data and model limits

The target has no default date or forecast horizon. Every account has twelve repayment records, and bureau snapshots have no reporting timestamp. Quality checks flag 3,875 collection events before loan origination, 7,293 loans before customer relationship start, and 13,143 bureau lines opened after the customer's latest loan. Preserve and disclose these inconsistencies; do not invent historical availability. Other plausibility warnings are recorded in the quality report.

Scores are **retrospective classifications, not validated future probabilities of default**. Customer-disjoint splits prevent entity overlap but cannot fix temporal leakage. IDs, gender and collections are excluded from model predictors. Missing DPD is unknown, not zero. Bureau summaries repeat across a customer's loans and must not be summed as unique bureau exposure. Anomaly scores measure unusualness, not default likelihood.

## Repository and requirement map

- `src/ingestion.py`, `src/features.py`: all-five-table ingestion, quality decisions, child aggregation, clean account dataset.
- `src/analytics.py`: metric definitions, dimensional profiles, five insights with business implications.
- `src/modeling.py`: grouped splits, logistic/boosting models, AUC/KS/Gini, importance and account drivers.
- `src/qa.py`: configurable Groq adapter, strict query schema, local execution, refusal and evidence.
- `src/anomalies.py`, `src/reviews.py`: unlabeled detectors, reasons, persistent versioned decisions.
- `app/app.py`: interactive portfolio/account views and both embedded AI workflows.
- `src/build.py`, `src/runtime.py`: reproducible artifacts, dependency/integrity checks, startup without training.
- `data/raw/`, `data/processed/`, `artifacts/`: immutable CSV copies, typed details/account data, models/splits/scores/reports/manifest.
- [Architecture PDF](docs/architecture.pdf) and [solution overview PDF](docs/solution_overview.pdf): the two one-page submission documents, generated from actual artifacts by `src/reporting.py`.
- [Documentation index](docs/README.md): approach, data/modeling, implementation specification, monitoring, and validation/submission details.
- `tests/`: join/reconciliation, grouped evaluation, grounded queries, provider failure/boundary, anomaly rules, persistence and app tests.

## Validation

Run `python -m pytest -q` after a build. Tests use actual synthetic artifacts plus small fault fixtures and Streamlit AppTest. Unit tests mock provider HTTP. Database tests exercise the real SQLite backend, idempotency, version history, ordering, reopen persistence, and CSV export; Streamlit tests cover saving decisions and reading them in a new session. All three live Groq checks passed against the configured `openai/gpt-oss-120b` model: full-portfolio figures, North-filtered figures, and unsupported future prediction refusal. The automated suite includes backend and UI regression tests (PyMuPDF emits upstream SWIG deprecation warnings). A repeated full build produced identical output hashes for data, models, scores, metrics, quality reports, and anomalies. The Streamlit HTTP health endpoint returned 200. `python -m src.smoke_llm` runs three opt-in credentialed checks (full scope, filtered scope, unsupported future question). Optional managed-Postgres connectivity remains unverified; no database URL is required for SQLite. Automated Streamlit interaction tests passed; visual browser verification was blocked by the browser automation tool.

Rebuild with `python -m src.build` after source/data changes. The manifest records input/output hashes, code hashes, package versions and seed. Repeated builds are checked for identical data, splits, scores, metrics and anomalies. `python -m src.reporting` regenerates just the PDFs from a validated existing build. PDF page count and visual layout are checked separately.

## Hosting and submission

**Demo video:** [Watch the LendScope walkthrough on Loom](https://www.loom.com/share/8e5cf64ccdb6403d81750181ed843a9a)

For Streamlit Community Cloud, push this repository including `data/raw/`, `data/processed/`, `artifacts/`, and the two PDFs to GitHub. Select `app/app.py` as entry point and Python 3.11. Install the checked-in requirements, configure the Groq secrets and `REVIEW_STORAGE="sqlite"` (no database provisioning required), then open the app anonymously and test both AI features. Hosted runtime has not been verified in this workspace.

Keep the app and API quota available for **14 days after submission**. Submit one email containing three links: hosted app, matching GitHub repository (or grant the named reviewer private access), and a **3–5 minute demo video** showing both views and both AI features. No zip attachment or slide deck. The two PDFs are each one page. See [submission checklist](docs/validation-and-submission.md) for demo timing, monitoring and production prerequisites. No email is sent by this application.

For a timed 3?5 minute recording, follow the [demo script](docs/demo-script.md). It includes account drivers, grounded Q&A with refusal, an anomaly decision, CSV export, model results and the storage limitation.

Implementation references: [Groq structured outputs](https://console.groq.com/docs/structured-outputs), [OpenAI-compatible structured query format](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat), and [Streamlit AppTest](https://docs.streamlit.io/develop/concepts/app-testing/get-started).
