"""Opt-in credentialed Groq/provider smoke test: python -m src.smoke_llm.

Makes three real API requests. Does not print keys or transmit row-level records.
"""
import os

from dotenv import load_dotenv

from src.analytics import grouped
from src.common import ROOT
from src.qa import LLMAdapter, execute_query
from src.runtime import load_artifacts


def main():
    load_dotenv(ROOT / ".env")
    data = load_artifacts()
    accounts = data["accounts"]
    settings = {k: os.getenv(k, "") for k in ["LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"]}
    adapter = LLMAdapter(settings)
    question = "Which loan type has the highest default rate?"
    for filters in [{}, {"region": ["North"]}]:
        query = adapter.plan_query(question, accounts, filters)
        result = execute_query(accounts, query, filters, data["manifest"]["build_id"])
        selected = accounts if not filters else accounts[accounts.region == "North"]
        expected = grouped(selected, ["loan_type"]).sort_values(["default_rate", "loan_type"], ascending=[False, True]).iloc[0]
        if not result["supported"] or not result.get("records"):
            raise ValueError("Provider did not support the basic aggregate question")
        first = result["records"][0]
        if first.get("loan_type") != expected.loan_type or abs(first.get("default_rate", -1) - expected.default_rate) > 1e-10:
            raise ValueError("Provider interpretation did not match the independently calculated answer")
        print("PASS", filters or "Full portfolio", "-", result["answer"])
    query = adapter.plan_query("What will the default rate be in the year 2030?", accounts, {})
    if query.supported:
        raise ValueError("Provider should decline future prediction from these data")
    print("PASS unsupported future question declined")


if __name__ == "__main__":
    main()
