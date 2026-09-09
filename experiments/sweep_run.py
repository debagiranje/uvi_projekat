"""sweep parametara po principu OFAT (one-factor-at-a-time)

za svaki parametar iz sweep liste u config-u, drzi ostale ACO parametre
fiksnim na base_aco vrijednostima i mijenja samo taj jedan parametar kroz
zadate vrijednosti, sa vise ponavljanja (n_repeats) po razlicitim seed-ovima
radi statisticke pouzdanosti (mean/std umjesto jednog run-a)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

# omogucava pokretanje skripte direktno (python experiments/run_sweep.py) bez
# potrebe da se projekat instalira ili PYTHONPATH rucno podesava
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aco.colony import run_aco
from src.tsp.instance import TSPInstance

DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "experiments" / "sweep_results.csv"

# redoslijed kolona u CSV-u - fiksan da bi generate_report.py mogao
# pouzdano da cita bez obzira koji je parametar trenutno sweep-ovan
CSV_COLUMNS = [
    "swept_parameter", "alpha", "beta", "rho", "q", "n_ants", "n_iterations",
    "repeat", "seed", "best_length", "runtime_seconds",
]


def load_sweep_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config file {path} mora da sadrzi YAML mapu (dict).")
    return config


def run_sweep(config: dict, output_csv: Path = DEFAULT_OUTPUT_CSV) -> list[dict]:
    """izvrsava sweep definisan u config-u, vraca listu rezultata i pise CSV

    config mora sadrzati: 'instance', 'base_aco' (dict fiksnih ACO
    parametara), 'sweep' listu params, opciono 'n_repeats' i 'seed'.
    """
    instance = TSPInstance.from_tsplib_file(config["instance"])
    base_aco = config["base_aco"]
    n_repeats = int(config.get("n_repeats", 3))
    base_seed = int(config.get("seed", 42))

    rows: list[dict] = []
    for dimension in config["sweep"]:
        param_name = dimension["parameter"]
        if param_name not in base_aco:
            raise ValueError(
                f"parametar '{param_name}' iz sweep liste nije definisan u base_aco"
            )

        for value in dimension["values"]:
            for repeat in range(n_repeats):
                aco_params = dict(base_aco)
                aco_params[param_name] = value
                seed = base_seed + repeat

                t0 = time.time()
                result = run_aco(
                    instance.distance_matrix,
                    n_ants=int(aco_params["n_ants"]),
                    n_iterations=int(aco_params["n_iterations"]),
                    alpha=float(aco_params["alpha"]),
                    beta=float(aco_params["beta"]),
                    rho=float(aco_params["rho"]),
                    q=float(aco_params["q"]),
                    seed=seed,
                )
                elapsed = time.time() - t0

                rows.append({
                    "swept_parameter": param_name,
                    "alpha": aco_params["alpha"],
                    "beta": aco_params["beta"],
                    "rho": aco_params["rho"],
                    "q": aco_params["q"],
                    "n_ants": aco_params["n_ants"],
                    "n_iterations": aco_params["n_iterations"],
                    "repeat": repeat,
                    "seed": seed,
                    "best_length": result.best_length,
                    "runtime_seconds": elapsed,
                })

    _write_csv(rows, output_csv)
    return rows


def _write_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        raise ValueError("nema rezultata za upis - provjeri sweep dio config-a")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="parameter sweep za ACO")
    parser.add_argument("--config", default="configs/sweep.yaml", help="putanja do sweep configaa")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV), help="putanja za CSV rezultate")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    t0 = time.time()
    rows = run_sweep(config, output_csv=Path(args.output))
    elapsed = time.time() - t0

    print(f"sweep zavrsen: {len(rows)} run-ova za {elapsed:.1f}s")
    print(f"rezultati sacuvani u: {args.output}")


if __name__ == "__main__":
    main()
