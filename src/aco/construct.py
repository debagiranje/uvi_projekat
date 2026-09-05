"""Konstrukcija TSP rute jednog mrava: pravilo prelaska ACO algoritma"""
from __future__ import annotations

import numpy as np


def construct_tour(
    distance_matrix: np.ndarray,
    pheromone: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    start_city: int | None = None,
) -> list[int]:
    """Konstruise jednu rutu (permutaciju svih gradova) za jednog mrava

    Vjerovatnoca prelaska iz grada i u neusmjeren grad j:
        p_ij = (tau_ij^alpha * eta_ij^beta) / sum_k (tau_ik^alpha * eta_ik^beta)
    gdje je eta_ij = 1 / distance_ij (heuristicka "vidljivost").

    Parametri:
        distance_matrix: (n, n) matrica udaljenosti
        pheromone: (n, n) matrica feromona (tau)
        alpha: uticaj feromona (veci alpha -> vise prati postojece feromone)
        beta: uticaj heuristike (veci beta -> vise prati kratke ivice)
        rng: numpy random generator (za reproducibilnost - proslijedi seeded rng)
        start_city: polazni grad; ako None, bira se nasumicno

    Vraca: listu indeksa gradova (permutacija), predstavlja jednu zatvorenu rutu
    """
    n = distance_matrix.shape[0]
    if n == 0:
        raise ValueError("distance_matrix ne smije biti prazna")
    if pheromone.shape != distance_matrix.shape:
        raise ValueError("pheromone i distance_matrix moraju imati isti oblik")

    # heuristicka vidljivost eta = 1/distance, sa 0 na dijagonali (grad sam sebi)
    with np.errstate(divide="ignore"):
        eta = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0.0)

    visited = np.zeros(n, dtype=bool)
    current = start_city if start_city is not None else rng.integers(0, n)
    tour = [int(current)]
    visited[current] = True

    for _ in range(n - 1):
        # probs samo ka nu gradovima
        weights = (pheromone[current] ** alpha) * (eta[current] ** beta)
        weights = np.where(visited, 0.0, weights)

        total = weights.sum()
        if total <= 0:
            # degenrisan slucaj (npr. svi feromoni = 0): biraj uniformno
            # nasumicno medju nu gradovima, umjesto pada u gresku
            candidates = np.where(~visited)[0]
            next_city = int(rng.choice(candidates))
        else:
            probabilities = weights / total
            next_city = int(rng.choice(n, p=probabilities))

        tour.append(next_city)
        visited[next_city] = True
        current = next_city

    return tour
