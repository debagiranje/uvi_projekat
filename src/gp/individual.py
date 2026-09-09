"""Enkofovanje GP jedinke: jedinka = vektor parametara ACO algoritma

GP ovdje ne zna nista o ACI ili TSPu - samo evoluira brojeve unutar
definisanih granica. Fitness funkcija (koja zna kako da poveze gene sa
stvarnim ACO runom) se dodaje spolja u evolve.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Opisuje jedan parametar koji GP evoluira: naziv i dozvoljeni opseg"""

    name: str
    low: float
    high: float

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"'{self.name}': low ({self.low}) mora biti < high ({self.high})")


# Podrazumijevani skup parametara ACE koje GP evoluira, sa opsezima
# preuzetim iz configs/*.yaml konvencije (alpha_range, beta_range, itd.)
DEFAULT_ACO_PARAM_SPECS: list[ParameterSpec] = [
    ParameterSpec("alpha", 0.5, 5.0),
    ParameterSpec("beta", 0.5, 5.0),
    ParameterSpec("rho", 0.01, 0.5),
    ParameterSpec("q", 1.0, 500.0),
    ParameterSpec("n_ants", 5.0, 100.0),
]


@dataclass
class Individual:
    """Jedna GP jedinka je vektor gena (u istom redoslijedu kao specs) + fitness."""

    genes: np.ndarray
    fitness: float | None = field(default=None)

    def as_dict(self, specs: list[ParameterSpec]) -> dict[str, float]:
        """Pretvara genski vektor u citljiv rjecnik {naziv_parametra: vrijednost}"""
        if len(self.genes) != len(specs):
            raise ValueError("Broj gena ne odgovara broju specifikacija parametara")
        return {spec.name: float(gene) for spec, gene in zip(specs, self.genes)}

    def copy(self) -> "Individual":
        return Individual(genes=self.genes.copy(), fitness=self.fitness)


def random_individual(specs: list[ParameterSpec], rng: np.random.Generator) -> Individual:
    """Generise jedinku sa genima nasumicno uniformno rasporedjenim unutar granica."""
    genes = np.array([rng.uniform(spec.low, spec.high) for spec in specs])
    return Individual(genes=genes)


def clip_to_bounds(genes: np.ndarray, specs: list[ParameterSpec]) -> np.ndarray:
    """Osigurava da svaki gen ostane unutar svog dozvoljenog opsega."""
    if len(genes) != len(specs):
        raise ValueError("Broj gena ne odgovara broju specifikacija parametara")
    low = np.array([s.low for s in specs])
    high = np.array([s.high for s in specs])
    return np.clip(genes, low, high)