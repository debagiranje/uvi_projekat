from pathlib import Path

import pytest

from src.pipeline import load_config, run

SMOKE_CONFIG = Path(__file__).parent.parent / "configs" / "smoke_test.yaml"


def test_load_config_reads_yaml():
    config = load_config(SMOKE_CONFIG)
    assert config["seed"] == 42
    assert "aco" in config
    assert "gp" in config


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nea.yaml")


def test_run_returns_expected_keys():
    config = load_config(SMOKE_CONFIG)
    result = run(config)
    assert set(result.keys()) == {"seed", "instance", "best_tour_length", "status"}
    assert result["seed"] == 42
