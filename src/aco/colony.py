"""ACO kolonija: upravlja sa N mrava kroz M iteracija, vraca najbolju rutu"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.aco.construct import construct_tour
from src.aco.pheromone import PheromoneMatrix


@dataclass
class ACOResult:
    """ rezultat jednog ACO run-a."""

    best_tour: list[int]
    best_length: float
    # duzina najbolje rute po iteraciji
    convergence: list[float] = field(default_factory=list)


def run_aco(
    distance_matrix: np.ndarray,
    n_ants: int,
    n_iterations: int,
    alpha: float,
    beta: float,
    rho: float,
    q: float,
    seed: int | None = None,
    initial_pheromone: float = 1.0,
) -> ACOResult:
    """pokrece ACO algoritam na datoj matrici udaljenosti

    parametri:
        distance_matrix: (n, n) matrica udaljenosti izmedju gradova
        n_ants: broj mrava po iteraciji
        n_iterations: broj iteracija algoritma
        alpha: uticaj feromona pri izboru sljedecg grada
        beta: uticaj heuristike (1/udaljenost) pri izboru sljedeceg grada
        rho: stopa isparavanja feromona, (0, 1)
        q: konstanta koja skalira depoziciju feromona (Q / duzina_rute)
        seed: random seed za reproducibilnost
        initial_pheromone: pocetna vrijednost feromona na svim ivicama

    Vraca: ACOResult sa najboljom rutom, njenom duzinom i istorijom
    konvergencije
    """
    if n_ants < 1:
        raise ValueError("n_ants mora biti bar 1")
    if n_iterations < 1:
        raise ValueError("n_iterations mora biti bar 1")

    n_cities = distance_matrix.shape[0]
    rng = np.random.default_rng(seed)
    pheromone_matrix = PheromoneMatrix(n_cities, initial_value=initial_pheromone)

    best_tour: list[int] | None = None
    best_length = float("inf")
    convergence: list[float] = []

    for _ in range(n_iterations):
        tours: list[list[int]] = []
        lengths: list[float] = []

        for _ in range(n_ants):
            tour = construct_tour(
                distance_matrix, pheromone_matrix.tau, alpha=alpha, beta=beta, rng=rng
            )
            length = _tour_length(distance_matrix, tour)
            tours.append(tour)
            lengths.append(length)

            if length < best_length:
                best_length = length
                best_tour = tour

        pheromone_matrix.evaporate(rho)
        pheromone_matrix.deposit_from_ants(tours, lengths, q)

        convergence.append(best_length)

    assert best_tour is not None  # garantovano jer n_ants >= 1 i n_iterations >= 1
    return ACOResult(best_tour=best_tour, best_length=best_length, convergence=convergence)


def _tour_length(distance_matrix: np.ndarray, tour: list[int]) -> float:
    n = len(tour)
    return sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
