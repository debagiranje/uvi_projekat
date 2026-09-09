"""Glavna GP petlja: evoluira populaciju jedinki (vektora parametara).

Namjerno ne zna nista o ACI ili TSPu - fitness_fn je proizvoljna
funkcija koju poziva spolja. Ovo omogucava nezavisno testiranje GP
logike (npr. sa jednostavnom matematickom funkcijom) prije full integracije sa ACOm
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.gp.individual import Individual, ParameterSpec, random_individual
from src.gp.operators import crossover, mutate, tournament_selection


@dataclass
class GPResult:
    """Rezultat GP evolucije"""

    best_individual: Individual
    # najbolji (najniži) fitness po generaciji - za analizu konvergencije
    best_fitness_history: list[float] = field(default_factory=list)


def run_gp(
    fitness_fn: Callable[[Individual], float],
    specs: list[ParameterSpec],
    population_size: int,
    n_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_size: int,
    seed: int | None = None,
    elitism: int = 1,
) -> GPResult:
    """Pokrece GP evoluciju koja MINIMIZUJE fitness_fn (nizi fitness = bolji).

    Parametri:
        fitness_fn: funkcija Individual -> float (npr. duzina najbolje ACO
            rute za parametre kodirane u toj jedinki)
        specs: opis parametara koji se evoluiraju (imena i granice)
        population_size: velicina populacije
        n_generations: broj generacija
        crossover_rate: vjerovatnoca da se par roditelja ukrsti
            (inace se kopiraju nepromijenjeni)
        mutation_rate: vjerovatnoca mutacije PO GENU
        tournament_size: broj jedinki u turnirskoj selekciji
        seed: random seed za reproducibilnost
        elitism: broj najboljih jedinki koje prelaze u sljedecu generaciju
            nepromijenjene (garantuje da se najbolje rjesenje nikad ne izgubi)

    Vraca: GPResult sa najboljom jedinkom i istorijom konvergencije
    """
    if population_size < 2:
        raise ValueError("population_size mora biti bar 2")
    if n_generations < 1:
        raise ValueError("n_generations mora biti bar 1")
    if elitism < 0 or elitism >= population_size:
        raise ValueError("elitism mora biti u opsegu [0, population_size)")

    rng = np.random.default_rng(seed)

    population = [random_individual(specs, rng) for _ in range(population_size)]
    for ind in population:
        ind.fitness = fitness_fn(ind)

    best_fitness_history: list[float] = []
    best_individual = min(population, key=lambda ind: ind.fitness).copy()

    for _ in range(n_generations):
        population.sort(key=lambda ind: ind.fitness)
        next_population = [ind.copy() for ind in population[:elitism]]

        while len(next_population) < population_size:
            parent1 = tournament_selection(population, tournament_size, rng)
            parent2 = tournament_selection(population, tournament_size, rng)

            if rng.uniform() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, specs, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = mutate(child1, specs, mutation_rate, rng)
            child2 = mutate(child2, specs, mutation_rate, rng)

            next_population.append(child1)
            if len(next_population) < population_size:
                next_population.append(child2)

        for ind in next_population:
            if ind.fitness is None:
                ind.fitness = fitness_fn(ind)

        population = next_population

        generation_best = min(population, key=lambda ind: ind.fitness)
        if generation_best.fitness < best_individual.fitness:
            best_individual = generation_best.copy()

        best_fitness_history.append(best_individual.fitness)

    return GPResult(best_individual=best_individual, best_fitness_history=best_fitness_history)