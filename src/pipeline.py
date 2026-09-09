"""
main entry point koji spaja GP i ACO slojeve

kako bi ovo trebalo da radi?

GP evoluira parametre ACO algoritma (alpha, beta, rho, q) dakle, oni koji imaju range sufix.
za svaku GP jedinku, fitness_fn dkodira njene gene u konkretne ACO parametre, pokrece pun ACO run
na zadatoj TSP instanci,
i vraca duzinu najbolje pronadjene rute kao fitness (GP minimizuje ovu vrijednost)

n_ants i n_iterations ostaju FIKSNI iz config-a, jer nisu predmet evvolulice
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from src.aco.colony import run_aco
from src.gp.evolve import run_gp
from src.gp.individual import Individual, ParameterSpec
from src.tsp.instance import TSPInstance


def load_config(path: str | Path) -> dict:
    """ucitava YAML konfiguraciju eksperimenta"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config fajl {path} mora sadrzati YAML mapu (dict).")
    return config


def build_param_specs(aco_config: dict) -> list[ParameterSpec]:
    """ gradi listu ParameterSpec objekata iz aco sekcije config-a.

    svaki kljuc koji iam range sufix postaje parametar koji GP evoluira
    """
    specs = []
    for key, value in aco_config.items():
        if key.endswith("_range"):
            name = key[: -len("_range")]
            low, high = value
            specs.append(ParameterSpec(name=name, low=float(low), high=float(high)))
    if not specs:
        raise ValueError(
            "aco config ne sadrzi nijedan range kljuc - nema sta da GP evoluira"
        )
    return specs


def make_fitness_fn(
    distance_matrix: np.ndarray,
    specs: list[ParameterSpec],
    n_ants: int,
    n_iterations: int,
    aco_seed_rng: np.random.Generator,
) -> Callable[[Individual], float]:
    """pravi fitness funkciju koja povezuje GP jedinku sa stvarnim ACO run-om.

    Svaki poziv dekodira gene jedinke u ACO parametre (alpha, beta, rho, q),
    pokrece run_aco s tim parametrima i fiksnim n_ants/n_iterations, i
    vraca duzinu najbolje pronasjene rute (nize = bolje, GP minimizuje).

    aco_seed_rng se koristi da bi svaki aco run deterministicki dobio svoj seed
    """

    def fitness_fn(individual: Individual) -> float:
        params = individual.as_dict(specs)
        aco_seed = int(aco_seed_rng.integers(0, 2**31 - 1))
        result = run_aco(
            distance_matrix,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=params["alpha"],
            beta=params["beta"],
            rho=params["rho"],
            q=params["q"],
            seed=aco_seed,
        )
        return result.best_length

    return fitness_fn


def run(config: dict) -> dict:
    """runnovanje punog pipelinea: GP evoluira ACO parametre na zadatoj TSP instanci

    Vraca dict sa najboljim pronadjenim parametrima, duzinom najbolje
    rute i istorijom konvergencije GP-a, analiza TBD
    """
    seed = config.get("seed")
    instance = TSPInstance.from_tsplib_file(config["instance"])

    aco_config = config["aco"]
    specs = build_param_specs(aco_config)
    n_ants = int(aco_config["n_ants"])
    n_iterations = int(aco_config["n_iterations"])

    aco_seed_rng = np.random.default_rng(seed)
    fitness_fn = make_fitness_fn(instance.distance_matrix, specs, n_ants, n_iterations, aco_seed_rng)

    gp_config = config["gp"]
    gp_result = run_gp(
        fitness_fn=fitness_fn,
        specs=specs,
        population_size=int(gp_config["population_size"]),
        n_generations=int(gp_config["n_generations"]),
        crossover_rate=float(gp_config["crossover_rate"]),
        mutation_rate=float(gp_config["mutation_rate"]),
        tournament_size=int(gp_config["tournament_size"]),
        seed=seed,
    )

    return {
        "seed": seed,
        "instance": instance.name,
        "best_tour_length": gp_result.best_individual.fitness,
        "best_params": gp_result.best_individual.as_dict(specs),
        "convergence": gp_result.best_fitness_history,
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ACO + GP za TSP")
    parser.add_argument("--config", required=True, help="Putanja do YAML config fajla")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run(config)
    print(result)


if __name__ == "__main__":
    main()
