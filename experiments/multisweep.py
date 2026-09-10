"""Poredi uticaj ACO parametara preko vise TSP instanci (generalizacija)"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CSV = Path(__file__).parent.parent / "experiments" / "sweep_results.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "comparison_results"

_COLOR_CYCLE = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#BA7517", "#7F77DD"]


def load_results(csv_path: str | Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_by_instance_and_value(
    rows: list[dict], param_name: str
) -> dict[str, dict[float, float]]:
    """vraca {instanca: {vrijednost_parametra: prosjecna_duzina}} za dati parametar"""
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["swept_parameter"] != param_name:
            continue
        instance = row["instance"]
        value = float(row[param_name])
        grouped[instance][value].append(float(row["best_length"]))

    if not grouped:
        raise ValueError(f"nema podataka za parametar '{param_name}' u rezultatima")

    return {
        instance: {value: statistics.mean(lengths) for value, lengths in by_value.items()}
        for instance, by_value in grouped.items()
    }


def normalize_relative_to_best(by_value: dict[float, float]) -> dict[float, float]:
    """dijeli svaku prosjecnu duzinu sa minimalnom nadjenom"""
    best = min(by_value.values())
    return {value: length / best for value, length in by_value.items()}


def plot_parameter_across_instances(
    param_name: str, per_instance: dict[str, dict[float, float]], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))

    for i, (instance, by_value) in enumerate(sorted(per_instance.items())):
        normalized = normalize_relative_to_best(by_value)
        values = sorted(normalized.keys())
        relative = [normalized[v] for v in values]
        color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
        ax.plot(values, relative, marker="o", label=instance, color=color, linewidth=1.5)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel("relativna duzina (1.0 = najbolje nadjeno za tu instancu)")
    ax.set_title(f"da li uticaj parametra '{param_name}' generalizuje preko instanci?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_report(
    csv_path: Path = DEFAULT_CSV, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> list[Path]:
    rows = load_results(csv_path)
    if not rows:
        raise ValueError(f"nema podataka u {csv_path}")
    if "instance" not in rows[0]:
        raise ValueError(
            "CSV ne sadrzi 'instance' kolonu - pokreni run_sweep.py sa 'instances' "
            "listom u config-u, ili koristi generate_report.py za sweep jedne instance."
        )

    distinct_instances = sorted({row["instance"] for row in rows})
    if len(distinct_instances) < 2:
        raise ValueError(
            f"CSV sadrzi samo jednu instancu ({distinct_instances}) "
        )

    swept_params = sorted({row["swept_parameter"] for row in rows})
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for param_name in swept_params:
        per_instance = summarize_by_instance_and_value(rows, param_name)
        output_path = output_dir / f"sweep_multi_instance_{param_name}.png"
        plot_parameter_across_instances(param_name, per_instance, output_path)
        generated_files.append(output_path)

        print(f"\n=== Parametar '{param_name}' - najbolja vrijednost po instanci ===")
        for instance in distinct_instances:
            by_value = per_instance.get(instance)
            if not by_value:
                continue
            best_value = min(by_value, key=by_value.get)
            print(f"  {instance}: najbolje pri {param_name}={best_value}")

    return generated_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="poredi uticaj ACO parametara preko vise TSP instanci"
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    files = generate_report(Path(args.csv), Path(args.output_dir))
    print(f"\ngenerisano {len(files)} grafova u {args.output_dir}/")


if __name__ == "__main__":
    main()