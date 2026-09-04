""" feromonska matrica: inicijalizacija, isparavanje i depozicija"""
from __future__ import annotations

import numpy as np


class PheromoneMatrix:
    """cuva i azurira nivoe feromona na svakoj ivici (i, j).

    Matrica je simetricna (tau[i][j] == tau[j][i]) jer posmatramo
    simetrican TSP (neusmjeren graf).
    """

    def __init__(self, n_cities: int, initial_value: float = 1.0):
        if initial_value < 0:
            raise ValueError("initial_value ne smije biti negativan")
        self.n_cities = n_cities
        self.tau = np.full((n_cities, n_cities), initial_value, dtype=float)
        np.fill_diagonal(self.tau, 0.0)

    def evaporate(self, rho: float) -> None:
        """isparavanje feromona: tau = (1 - rho) * tau.

        rho iz (0, 1) - udio feromona koji isparava svake iteracije
        """
        if not (0.0 < rho < 1.0):
            raise ValueError("rho mora biti u opsegu (0, 1)")
        self.tau *= 1.0 - rho
        np.fill_diagonal(self.tau, 0.0)

    def deposit(self, tour: list[int], amount: float) -> None:
        """Dodaje feromon duz ivica jedne rute (simetricno na oba smjera)."""
        n = len(tour)
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            self.tau[a, b] += amount
            self.tau[b, a] += amount

    def deposit_from_ants(
        self, tours: list[list[int]], tour_lengths: list[float], q: float
    ) -> None:
        """ pheromone deposition za sve mrave odjednom: delta_tau = Q / duzina_rute

        Krace rute (bolja rjesenja) dobijaju veci deopsit
        ovo je mehanizam pozitivne povratne sprege ka boljim rutama
        """
        for tour, length in zip(tours, tour_lengths):
            if length <= 0:
                raise ValueError("duzina rute mora biti pozitivna")
            self.deposit(tour, amount=q / length)
