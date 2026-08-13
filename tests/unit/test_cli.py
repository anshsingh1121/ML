"""Unit tests for Enterprise CLI (`main.py` & `src/cli/main_cli.py`)."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.cli.main_cli import EnterpriseCLI
from main import main


@pytest.fixture
def cli_engine() -> EnterpriseCLI:
    """Create EnterpriseCLI instance."""
    return EnterpriseCLI()


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    """Generate minimal synthetic CSV for CLI command verification."""
    csv_path = str(tmp_path / "test_incidents.csv")
    df = pd.DataFrame({
        "number": [f"INC00000{i}" for i in range(1, 16)],
        "opened_at": [f"2025-01-0{i%9+1} 10:00:00" for i in range(1, 16)],
        "resolved_at": [f"2025-01-0{i%9+1} 12:00:00" for i in range(1, 16)],
        "priority": [1, 2, 3, 2, 1] * 3,
        "business_impact": [1, 2, 2, 1, 1] * 3,
        "urgency": [1, 2, 2, 1, 1] * 3,
        "category": ["Network", "Database", "Software", "Hardware", "Network"] * 3,
        "assignment_group": ["Network Support", "Database Support", "App Support", "Hardware Support", "Network Support"] * 3,
        "short_description": ["Server network port connection lost"] * 15,
        "description": ["Router switch interface gigabit down after maintenance"] * 15,
        "cmdb_ci": ["ROUTER-01"] * 15,
        "business_service": ["Core Banking"] * 15,
        "reassignment_count": [0, 1, 0, 2, 0] * 3,
        "reopen_count": [0] * 15,
        "made_sla": [True, True, False, True, True] * 3,
        "sla_status": ["Met", "Met", "Breached", "Met", "Met"] * 3
    })
    df.to_csv(csv_path, index=False)
    return csv_path


def test_cli_status(cli_engine: EnterpriseCLI) -> None:
    """Test cmd_status output."""
    res = cli_engine.cmd_status()
    assert res == 0


def test_cli_validate(cli_engine: EnterpriseCLI, sample_csv: str) -> None:
    """Test cmd_validate command."""
    res_val = cli_engine.cmd_validate(input_path=sample_csv)
    assert res_val in (0, 2)  # 0 for PASS, 2 for warnings/failures


def test_cli_readiness(cli_engine: EnterpriseCLI, sample_csv: str) -> None:
    """Test cmd_readiness command."""
    res = cli_engine.cmd_readiness(input_path=sample_csv)
    assert res in (0, 2)


def test_cli_eda(cli_engine: EnterpriseCLI, sample_csv: str, tmp_path: Path) -> None:
    """Test cmd_eda command."""
    out_dir = str(tmp_path / "eda_reports")
    res = cli_engine.cmd_eda(input_path=sample_csv, output_dir=out_dir)
    assert res == 0
    assert (Path(out_dir) / "eda_report.json").exists()


def test_cli_clean_engineer_split(cli_engine: EnterpriseCLI, sample_csv: str, tmp_path: Path) -> None:
    """Test modular clean -> engineer -> split commands."""
    clean_path = str(tmp_path / "cleaned.csv")
    eng_path = str(tmp_path / "engineered.csv")
    split_dir = str(tmp_path / "splits")

    assert cli_engine.cmd_clean(input_path=sample_csv, output_path=clean_path) == 0
    assert Path(clean_path).exists()

    assert cli_engine.cmd_engineer(input_path=clean_path, output_path=eng_path) == 0
    assert Path(eng_path).exists()

    assert cli_engine.cmd_split(input_path=eng_path, strategy="stratified", target="assignment_group", output_dir=split_dir) == 0
    assert (Path(split_dir) / "train.csv").exists()


def test_cli_pipeline(cli_engine: EnterpriseCLI, sample_csv: str, tmp_path: Path) -> None:
    """Test end-to-end cmd_pipeline execution."""
    out_dir = str(tmp_path / "pipe_out")
    res = cli_engine.cmd_pipeline(input_path=sample_csv, output_dir=out_dir)
    assert res == 0
    assert (Path(out_dir) / "train.csv").exists()
    assert (Path(out_dir) / "val.csv").exists()
    assert (Path(out_dir) / "test.csv").exists()
    assert (Path(out_dir) / "master_engineered_incidents.csv").exists()


def test_main_argparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main.py command line argument parsing for status command."""
    monkeypatch.setattr("sys.argv", ["main.py", "status"])
    assert main() == 0


def test_cli_models_and_clean(cli_engine: EnterpriseCLI) -> None:
    """Test cmd_models and cmd_clean_workspace execution."""
    assert cli_engine.cmd_models() == 0
    assert cli_engine.cmd_clean_workspace() == 0


def test_cli_run_command(cli_engine: EnterpriseCLI) -> None:
    """Test run_command dispatch across various subcommands."""
    class DummyArgs:
        def __init__(self, cmd: str, **kwargs: object) -> None:
            self.command = cmd
            for k, v in kwargs.items():
                setattr(self, k, v)

    assert cli_engine.run_command(DummyArgs("status")) == 0
    assert cli_engine.run_command(DummyArgs("models")) == 0
    assert cli_engine.run_command(DummyArgs("clean-workspace")) == 0
    assert cli_engine.run_command(DummyArgs("unknown_command")) == 1


def test_cli_menu_noninteractive(cli_engine: EnterpriseCLI, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test non-interactive terminal menu fallback."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli_engine.run_interactive_menu() == 0

