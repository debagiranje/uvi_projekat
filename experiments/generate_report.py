"""generise grafove uticaja ACO parametara iz sweep_results.csv 

za svaki parametar koji je bio sweepovan, racuna prosjecnu duzinu rute
i standardnu devijaciju po vrijednosti parametra (preko ponavljanja sa
razlicitim seedovima), i crta grafik zavisnosti kvaliteta rjesenja od
vrijednosti parametra.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  

DEFAULT_CSV = Path(__file__).parent.parent / "experiments" / "sweep_results.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "results"


def load_results(csv_path: str | Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_parameter(rows: list[dict], param_name: str) -> dict[float, tuple[float, float]]:
    """grupise rezultate po vrijednosti datog parametra.

    vraca {val: (prosjecna_duzina, std_devijacija)}, racunato preko
    svih ponavljanja (repeats) zabiljezenih za tu vrijednost
    """
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["swept_parameter"] != param_name:
            continue
        value = float(row[param_name])
        grouped[value].append(float(row["best_length"]))

    if not grouped:
        raise ValueError(f"nema podataka za parametar '{param_name}' u rezultatima")

    summary = {}
    for value, lengths in grouped.items():
        mean = statistics.mean(lengths)
        std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        summary[value] = (mean, std)
    return summary


def plot_parameter_influence(
    summary: dict[float, tuple[float, float]], param_name: str, output_path: Path
) -> None:
    values = sorted(summary.keys())
    means = [summary[v][0] for v in values]
    stds = [summary[v][1] for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(values, means, yerr=stds, marker="o", capsize=4, linewidth=1.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel("prosjecna duzina rute (best_length)")
    ax.set_title(f"uticaj parametra '{param_name}' na kvalitet rjesenja")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_report(
    csv_path: Path = DEFAULT_CSV, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[Path]:
    """generise po jedan graf za svaki sweepovan parametar. Vraca listu putanja."""
    rows = load_results(csv_path)
    if not rows:
        raise ValueError(f"nema podataka u {csv_path}")

    swept_params = sorted({row["swept_parameter"] for row in rows})
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for param_name in swept_params:
        summary = summarize_parameter(rows, param_name)
        output_path = output_dir / f"sweep_{param_name}.png"
        plot_parameter_influence(summary, param_name, output_path)
        generated_files.append(output_path)

        print(f"\n=== uticaj parametra '{param_name}' ===")
        for value in sorted(summary.keys()):
            mean, std = summary[value]
            print(f"  {param_name} = {value}: prosjecna duzina = {mean:.2f} (std={std:.2f})")

    return generated_files


def main() -> None:
    parser = argparse.ArgumentParser(description="generise izvjestaj o uticaju parametara")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    files = generate_report(Path(args.csv), Path(args.output_dir))
    print(f"\ngenerisano {len(files)} grafova u {args.output_dir}/")


if __name__ == "__main__":
    main()
