"""Unit tests for Enterprise Text Preprocessor (`src/preprocessing/text_preprocessor.py`)."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.preprocessing.text_preprocessor import TextPreprocessor


@pytest.fixture
def raw_text_df() -> pd.DataFrame:
    """Create synthetic text dataframe with stopwords, HTML, and protected keywords."""
    return pd.DataFrame({
        "number": ["INC000001", "INC000002", "INC000003"],
        "short_description": [
            "The server is down due to a critical network timeout!",
            "<html>Please kindly verify that the database login is slow and failing.</html>",
            "Not Provided"
        ],
        "description": [
            "From: admin@firstcitizens.com\nTo: support@firstcitizens.com\nSubject: Crash\n\n[System Error Code: 0x80040154] Server crashed after memory deadlock.",
            "We have reported that routers and switches are experiencing packet loss across connections.",
            None
        ]
    })


def test_text_preprocessor_initialization() -> None:
    """Test initialization of TextPreprocessor."""
    preprocessor = TextPreprocessor()
    assert preprocessor.max_seq_tokens == 256
    assert "server" not in preprocessor.stopwords
    assert "deadlock" not in preprocessor.stopwords
    assert "the" in preprocessor.stopwords


def test_text_preprocessor_normalize_text() -> None:
    """Test individual string normalization steps."""
    preprocessor = TextPreprocessor()

    # 1. Check stopword removal + protected keyword preservation
    txt1 = "The server is down and connection failed"
    clean1 = preprocessor.normalize_text(txt1)
    assert "the" not in clean1
    assert "is" not in clean1
    assert "server" in clean1
    assert "down" in clean1
    assert "connection" in clean1

    # 2. Check HTML and email stripping
    txt2 = "<html>From: test@test.com\nServer failure on router</html>"
    clean2 = preprocessor.normalize_text(txt2)
    assert "<html>" not in clean2
    assert "test@test.com" not in clean2
    assert "server" in clean2
    assert "failure" in clean2

    # 3. Check lemmatization
    txt3 = "failures on servers and routers"
    clean3 = preprocessor.normalize_text(txt3, lemmatize=True)
    assert "failure" in clean3
    assert "server" in clean3
    assert "router" in clean3


def test_text_preprocessor_pipeline(raw_text_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test full dataset text preprocessing and report generation."""
    preprocessor = TextPreprocessor()
    output_dir = str(tmp_path / "reports")

    proc_df, audit_report = preprocessor.preprocess_dataset(df=raw_text_df, output_dir=output_dir)

    # Verify new _clean columns created
    assert "short_description_clean" in proc_df.columns
    assert "description_clean" in proc_df.columns

    # Verify Not Provided handled
    assert proc_df.loc[2, "short_description_clean"] == "not provided"

    # Verify reports generated
    assert (Path(output_dir) / "text_preprocessing_report.json").exists()
    assert (Path(output_dir) / "text_preprocessing_report.md").exists()

    with open(Path(output_dir) / "text_preprocessing_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["status"] == "CERTIFIED_TEXT_CLEAN"
        assert "short_description" in data["column_metrics"]
