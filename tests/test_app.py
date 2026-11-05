import pytest
from streamlit.testing.v1 import AppTest

from src.common import ROOT


@pytest.fixture(autouse=True)
def isolated_review_store(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_STORAGE", "sqlite")
    monkeypatch.setenv("REVIEW_SQLITE_PATH", str(tmp_path / "reviews.db"))


def test_all_views_and_filters(monkeypatch):
    for key in ["LLM_API_KEY", "LLM_MODEL", "DATABASE_URL"]:
        monkeypatch.setenv(key, "")
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "14,685"
    app.sidebar.multiselect[0].set_value(["South"]).run()
    assert not app.exception and app.metric[0].value == "3,600"
    for view in ["Account Detail", "Ask the Portfolio", "Anomaly Review"]:
        app.sidebar.radio[0].set_value(view).run()
        assert not app.exception, view
    assert any("SQLite assessment mode" in x.value for x in app.warning)
    app.sidebar.radio[0].set_value("Portfolio").run()
    next(b for b in app.sidebar.button if b.label == "Clear filters").click().run()
    assert not app.exception and app.metric[0].value == "14,685"


def test_missing_llm_disables_submission(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Ask the Portfolio").run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "Ask portfolio").disabled


def test_question_submission_and_scope_change(monkeypatch):
    from src.qa import LLMAdapter, QueryPlan
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(LLMAdapter, "plan_query", lambda *args: QueryPlan(supported=True, reason="Highest observed default rate by product", metric="default_rate", group_by=["loan_type"], filters=[], sort="descending", limit=1))
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Ask the Portfolio").run()
    app.text_area[0].set_value("Which loan type has the highest default rate?")
    next(b for b in app.button if b.label == "Ask portfolio").click().run()
    assert not app.exception
    assert any("353 defaults / 1,432 labeled accounts" in item.value for item in app.markdown)
    app.sidebar.multiselect[0].set_value(["North"]).run()
    assert not app.exception
    assert not any("353 defaults / 1,432 labeled accounts" in item.value for item in app.markdown)


def test_sqlite_review_save_survives_new_session():
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Anomaly Review").run()
    assert not app.exception
    assert not next(b for b in app.button if b.label == "Save decision").disabled
    next(r for r in app.radio if r.label == "Decision").set_value("dismiss")
    app.text_area[0].set_value("Assessment demo: evidence reviewed")
    next(b for b in app.button if b.label == "Save decision").click().run()
    assert not app.exception
    assert any("Saved dismiss" in item.value for item in app.success)
    fresh = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
    fresh.sidebar.radio[0].set_value("Anomaly Review").run()
    assert not fresh.exception
    assert fresh.metric[2].value == "0 / 1"
    assert any("CSV contains 1 committed events" in item.value for item in fresh.caption)


def test_optional_postgres_missing_url_disables_save(monkeypatch):
    monkeypatch.setenv("REVIEW_STORAGE", "postgres")
    monkeypatch.setenv("DATABASE_URL", "")
    app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Anomaly Review").run()
    assert not app.exception
    assert next(b for b in app.button if b.label == "Save decision").disabled
    assert any("Review storage unavailable" in item.value for item in app.error)
