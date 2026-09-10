import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sweep_run import run_sweep  # noqa: E402

BURMA14 = Path(__file__).parent.parent / "data" / "instances" / "burma14.tsp"


def _tiny_config(**overrides):
    config = {
        "seed": 1,
        "instance": str(BURMA14),
        "n_repeats": 2,
        "base_aco": {
            "alpha": 1.0, "beta": 3.0, "rho": 0.3, "q": 100.0,
            "n_ants": 5, "n_iterations": 5,
        },
        "sweep": [{"parameter": "rho", "values": [0.1, 0.3]}],
    }
    config.update(overrides)
    return config


def test_run_sweep_produces_expected_number_of_rows(tmp_path):
    config = _tiny_config()
    rows = run_sweep(config, output_csv=tmp_path / "results.csv")
    # 2 vrijednosti rho * 2 repeats = 4 redova
    assert len(rows) == 4


def test_run_sweep_writes_valid_csv(tmp_path):
    config = _tiny_config()
    output_csv = tmp_path / "results.csv"
    run_sweep(config, output_csv=output_csv)

    assert output_csv.exists()
    with open(output_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert set(rows[0].keys()) == {
        "swept_parameter", "alpha", "beta", "rho", "q", "n_ants", "n_iterations",
        "repeat", "seed", "best_length", "runtime_seconds",
    }


def test_run_sweep_varies_only_swept_parameter(tmp_path):
    config = _tiny_config()
    rows = run_sweep(config, output_csv=tmp_path / "results.csv")
    # ostali parametri (osim rho) moraju ostati fiksni na base_aco vrijednosti
    for row in rows:
        assert row["alpha"] == 1.0
        assert row["beta"] == 3.0
        assert row["n_ants"] == 5


def test_run_sweep_unknown_parameter_raises(tmp_path):
    config = _tiny_config(sweep=[{"parameter": "does_not_exist", "values": [1, 2]}])
    with pytest.raises(ValueError):
        run_sweep(config, output_csv=tmp_path / "results.csv")


def test_run_sweep_deterministic_with_same_seed(tmp_path):
    config = _tiny_config()
    rows_a = run_sweep(config, output_csv=tmp_path / "a.csv")
    rows_b = run_sweep(config, output_csv=tmp_path / "b.csv")
    lengths_a = [r["best_length"] for r in rows_a]
    lengths_b = [r["best_length"] for r in rows_b]
    assert lengths_a == lengths_b
