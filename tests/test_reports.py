import fitz

from src.common import ROOT


def test_submission_pdfs_are_single_page_with_expected_content():
    for name in ["architecture", "solution_overview"]:
        with fitz.open(ROOT / "docs" / f"{name}.pdf") as document:
            assert len(document) == 1
            text = document[0].get_text()
            assert "Retrospective" in text or "retrospective" in text
            assert "GROQ" in text or "Groq" in text
            assert "SQLite" in text or "SQLITE" in text
            assert "CSV" in text
            for block in document[0].get_text("blocks"):
                assert block[0] >= 0 and block[1] >= 0
                assert block[2] <= document[0].rect.width and block[3] <= document[0].rect.height
