"""runnuje pun GP+ACO pipeline preko vise razlicitih TSP instanci.

ovo je pokusaj generalizacije pristupa na N razlicitih instanci, razlicitih velicina.
za svaku instancu se pokrece pipeline N puta (razliciti seedovi), i rezultat se poredi sa
poznatim TSPLIB optimumom
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

# omogucava pokretanje skripte direktno
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run as run_pipeline  # noqa: E402
from src.tsp.instance import TSPInstance  # noqa: E402

from data.know_optima import KNOWN_OPTIMA  # noqa: E402

DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "experiments" / "multi_instance_results.csv"

CSV_COLUMNS = [
    "instance", "n_cities", "repeat", "seed", "best_tour_length",
    "known_optimum", "gap_percent", "runtime_seconds",
]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config fajl {path} mora sadrzati YAML mapu (dict).")
    return config


def run_multi_instance(config: dict, output_csv: Path = DEFAULT_OUTPUT_CSV) -> list[dict]:
    """config mora sadrzati: 'instances' (lista putanja do .tsp fajlova),
    opciono 'n_repeats' i 'seed', i 'aco'/'gp' sekcije
    """
    instances = config["instances"]
    if not instances:
        raise ValueError("'instances' lista ne smije biti prazna")

    n_repeats = int(config.get("n_repeats", 3))
    base_seed = int(config.get("seed", 42))

    rows: list[dict] = []
    for instance_path in instances:
        instance_name = Path(instance_path).stem
        known_optimum = KNOWN_OPTIMA.get(instance_name)
        n_cities = TSPInstance.from_tsplib_file(instance_path).n_cities

        for repeat in range(n_repeats):
            seed = base_seed + repeat
            run_config = {
                "seed": seed,
                "instance": instance_path,
                "aco": config["aco"],
                "gp": config["gp"],
            }

            t0 = time.time()
            result = run_pipeline(run_config)
            elapsed = time.time() - t0

            best_length = result["best_tour_length"]
            gap_percent = (
                (best_length - known_optimum) / known_optimum * 100
                if known_optimum is not None else None
            )

            rows.append({
                "instance": instance_name,
                "n_cities": n_cities,
                "repeat": repeat,
                "seed": seed,
                "best_tour_length": round(best_length, 2),
                "known_optimum": known_optimum if known_optimum is not None else "",
                "gap_percent": round(gap_percent, 3) if gap_percent is not None else "",
                "runtime_seconds": round(elapsed, 2),
            })

            gap_str = f", gap={gap_percent:.2f}%" if gap_percent is not None else ""
            print(
                f"[{instance_name} ({n_cities} gradova)] repeat {repeat}: "
                f"duzina={best_length:.1f}{gap_str}, vrijeme={elapsed:.1f}s"
            )

    _write_csv(rows, output_csv)
    return rows


def _write_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        raise ValueError("nema rezultata za upis")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="multi-instance GP+ACO eksperiment")
    parser.add_argument("--config", default="configs/multi_instance.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    config = load_config(args.config)
    t0 = time.time()
    rows = run_multi_instance(config, output_csv=Path(args.output))
    elapsed = time.time() - t0

    print(f"\ngotovo: {len(rows)} run-ova za {elapsed:.1f}s")
    print(f"rezultati sacuvani u: {args.output}")


if __name__ == "__main__":
    main()
