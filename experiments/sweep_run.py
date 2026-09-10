"""sweep parametara po principu OFAT (one-factor-at-a-time)

za svaki parametar iz sweep liste u config-u, drzi ostale ACO parametre
fiksnim na base_aco vrijednostima i mijenja samo taj jedan parametar kroz
zadate vrijednosti, sa vise ponavljanja (n_repeats) po razlicitim seed-ovima
radi statisticke pouzdanosti (mean/std umjesto jednog run-a)

Podrzava 1 instancu ('instance: putanja.tsp') u config-u, ili vise instanci
odjednom ('instances: [putanja1.tsp, ...]'), radi provjere generalizacije uticaja params.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aco.colony import run_aco  # noqa: E402
from src.tsp.instance import TSPInstance  # noqa: E402

DEFAULT_OUTPUT_CSV = Path(__file__).parent.parent / "experiments" / "sweep_results.csv"

CSV_COLUMNS = [
    "instance", "n_cities", "swept_parameter", "alpha", "beta", "rho", "q",
    "n_ants", "n_iterations", "repeat", "seed", "best_length", "runtime_seconds",
]


def load_sweep_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config fajl {path} mora sadrzavati YAML")
    return config


def _get_instance_paths(config: dict) -> list[str]:
    """note, sad podrzava multisweep sa vise instanci iz configa"""
    if "instances" in config:
        instances = config["instances"]
        if not instances:
            raise ValueError("'instances' lista ne smije biti prazna")
        return instances
    if "instance" in config:
        return [config["instance"]]
    raise ValueError("config mora sadrzati 'instance' ili 'instances'")


def run_sweep(config: dict, output_csv: Path = DEFAULT_OUTPUT_CSV) -> list[dict]:
    """izvrsava sweep definisan u config-u, vraca listu rezultata i pise CSV """
    instance_paths = _get_instance_paths(config)
    base_aco = config["base_aco"]
    n_repeats = int(config.get("n_repeats", 3))
    base_seed = int(config.get("seed", 42))

    rows: list[dict] = []
    for instance_path in instance_paths:
        instance = TSPInstance.from_tsplib_file(instance_path)
        instance_name = Path(instance_path).stem

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
                        "instance": instance_name,
                        "n_cities": instance.n_cities,
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
        raise ValueError("nema rezultata za upis - provjeri sweep sekciju configa")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="parameter sweep za ACO")
    parser.add_argument("--config", default="configs/msweep.yaml", help="putanja do sweep configa")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV), help="putanja za CSV rezultate")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    t0 = time.time()
    rows = run_sweep(config, output_csv=Path(args.output))
    elapsed = time.time() - t0

    n_instances = len({r["instance"] for r in rows})
    print(f"sweep zavrsen: {len(rows)} runova preko {n_instances} instanci za {elapsed:.1f}s")
    print(f"rezultati sacuvani u: {args.output}")


if __name__ == "__main__":
    main()