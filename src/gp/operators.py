"""GP operatori: selekcija, crossover, mutacija nad jedinkama (vektorima parametara)"""
from __future__ import annotations

import numpy as np

from src.gp.individual import Individual, ParameterSpec, clip_to_bounds


def tournament_selection(
    population: list[Individual], tournament_size: int, rng: np.random.Generator
) -> Individual:
    """Bira jednog roditelja turnirskom selekcijom (minimizacija fitnessa)

    Nasumicno bira tournament_size jedinki i vraca onu sa najnizim
    (najboljim) fitnessom. Sve jedinke moraju imati postavljen fitness
    """
    if tournament_size < 1:
        raise ValueError("tournament_size mora biti bar 1")
    if tournament_size > len(population):
        raise ValueError("tournament_size ne smije biti veci od velicine populacije")

    indices = rng.choice(len(population), size=tournament_size, replace=False)
    contestants = [population[i] for i in indices]

    for c in contestants:
        if c.fitness is None:
            raise ValueError("Sve jedinke u turniru moraju imati postavljen fitness")

    return min(contestants, key=lambda ind: ind.fitness).copy()


def crossover(
    parent1: Individual,
    parent2: Individual,
    specs: list[ParameterSpec],
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> tuple[Individual, Individual]:
    """Aritmeticko (blend) ukrstanje: dijete = alpha*roditelj1 + (1-alpha)*roditelj2,
    i obrnuto za drugo dijete. Rezultat se ogranicva na dozvoljene granice
    """
    g1, g2 = parent1.genes, parent2.genes
    child1_genes = alpha * g1 + (1 - alpha) * g2
    child2_genes = alpha * g2 + (1 - alpha) * g1

    child1 = Individual(genes=clip_to_bounds(child1_genes, specs))
    child2 = Individual(genes=clip_to_bounds(child2_genes, specs))
    return child1, child2


def mutate(
    individual: Individual,
    specs: list[ParameterSpec],
    mutation_rate: float,
    rng: np.random.Generator,
    sigma_fraction: float = 0.1,
) -> Individual:
    """Gaussova mutacija: svaki gen se, sa vjerovatnocom mutation_rate,
    poremeti za N(0, sigma) gdje je sigma = sigma_fraction * (high - low)
    tog parametra. Rezultat se ogranicava na dozvoljene granice

    Vraca NOVU jedinku (ne mijenja infividual inplace)
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate mora biti u opsegu [0, 1]")

    genes = individual.genes.copy()
    for i, spec in enumerate(specs):
        if rng.uniform() < mutation_rate:
            sigma = sigma_fraction * (spec.high - spec.low)
            genes[i] += rng.normal(0, sigma)

    genes = clip_to_bounds(genes, specs)
    return Individual(genes=genes)