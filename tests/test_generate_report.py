import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_report import generate_report, load_results, summarize_parameter  # noqa: E402

CSV_COLUMNS = [
    "swept_parameter", "alpha", "beta", "rho", "q", "n_ants", "n_iterations",
    "repeat", "seed", "best_length", "runtime_seconds",
]


def _write_fake_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _fake_row(swept_parameter, rho, best_length, repeat=0):
    return {
        "swept_parameter": swept_parameter, "alpha": 1.0, "beta": 3.0,
        "rho": rho, "q": 100.0, "n_ants": 10, "n_iterations": 20,
        "repeat": repeat, "seed": repeat, "best_length": best_length,
        "runtime_seconds": 0.01,
    }


def test_summarize_parameter_computes_mean_and_std():
    rows = [
        _fake_row("rho", rho=0.1, best_length=100.0, repeat=0),
        _fake_row("rho", rho=0.1, best_length=110.0, repeat=1),
        _fake_row("rho", rho=0.3, best_length=200.0, repeat=0),
    ]
    summary = summarize_parameter(rows, "rho")

    assert summary[0.1][0] == pytest.approx(105.0)  # mean
    assert summary[0.1][1] > 0  # std > 0 (dvije razlicite vrijednosti)
    assert summary[0.3][0] == pytest.approx(200.0)
    assert summary[0.3][1] == pytest.approx(0.0)  # std = 0 (samo jedna vrijednost)


def test_summarize_parameter_missing_parameter_raises():
    rows = [_fake_row("rho", rho=0.1, best_length=100.0)]
    with pytest.raises(ValueError):
        summarize_parameter(rows, "does_not_exist")


def test_load_results_reads_csv(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_fake_csv(csv_path, [_fake_row("rho", 0.1, 100.0)])
    rows = load_results(csv_path)
    assert len(rows) == 1
    assert rows[0]["swept_parameter"] == "rho"


def test_generate_report_creates_one_png_per_swept_parameter(tmp_path):
    csv_path = tmp_path / "results.csv"
    rows = [
        _fake_row("rho", 0.1, 100.0, repeat=0),
        _fake_row("rho", 0.3, 110.0, repeat=0),
        _fake_row("n_ants", 5, 120.0, repeat=0),
    ]
    _write_fake_csv(csv_path, rows)

    output_dir = tmp_path / "reports"
    generated = generate_report(csv_path, output_dir)

    assert len(generated) == 2  # rho i n_ants
    for path in generated:
        assert path.exists()
        assert path.suffix == ".png"


def test_generate_report_empty_csv_raises(tmp_path):
    csv_path = tmp_path / "empty.csv"
    _write_fake_csv(csv_path, [])
    with pytest.raises(ValueError):
        generate_report(csv_path, tmp_path / "reports")
