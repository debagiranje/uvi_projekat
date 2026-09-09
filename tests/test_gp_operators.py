import numpy as np
import pytest

from src.gp.individual import Individual, ParameterSpec
from src.gp.operators import crossover, mutate, tournament_selection

SPECS = [ParameterSpec("a", 0.0, 10.0), ParameterSpec("b", 0.0, 10.0)]


def test_tournament_selection_picks_best_fitness():
    population = [
        Individual(genes=np.array([1.0, 1.0]), fitness=5.0),
        Individual(genes=np.array([2.0, 2.0]), fitness=1.0),  # najbolji (najniyi)
        Individual(genes=np.array([3.0, 3.0]), fitness=9.0),
    ]
    rng = np.random.default_rng(0)
    # tournament_size = broj jedinki u populaciji pa uvijek bira apsolutno najboljeg
    winner = tournament_selection(population, tournament_size=3, rng=rng)
    assert winner.fitness == pytest.approx(1.0)


def test_tournament_selection_returns_copy_not_reference():
    population = [Individual(genes=np.array([1.0]), fitness=1.0)]
    rng = np.random.default_rng(0)
    winner = tournament_selection(population, tournament_size=1, rng=rng)
    winner.genes[0] = 999.0
    assert population[0].genes[0] == pytest.approx(1.0)


def test_tournament_selection_missing_fitness_raises():
    population = [Individual(genes=np.array([1.0]), fitness=None)]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        tournament_selection(population, tournament_size=1, rng=rng)


def test_tournament_selection_invalid_size_raises():
    population = [Individual(genes=np.array([1.0]), fitness=1.0)]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        tournament_selection(population, tournament_size=0, rng=rng)
    with pytest.raises(ValueError):
        tournament_selection(population, tournament_size=5, rng=rng)


def test_crossover_children_within_bounds():
    p1 = Individual(genes=np.array([1.0, 9.0]))
    p2 = Individual(genes=np.array([9.0, 1.0]))
    rng = np.random.default_rng(0)

    c1, c2 = crossover(p1, p2, SPECS, rng)

    for gene, spec in zip(c1.genes, SPECS):
        assert spec.low <= gene <= spec.high
    for gene, spec in zip(c2.genes, SPECS):
        assert spec.low <= gene <= spec.high


def test_crossover_alpha_one_returns_parents_unchanged():
    p1 = Individual(genes=np.array([2.0, 3.0]))
    p2 = Individual(genes=np.array([7.0, 8.0]))
    rng = np.random.default_rng(0)

    c1, c2 = crossover(p1, p2, SPECS, rng, alpha=1.0)

    assert np.allclose(c1.genes, p1.genes)
    assert np.allclose(c2.genes, p2.genes)


def test_mutate_zero_rate_returns_unchanged_genes():
    ind = Individual(genes=np.array([5.0, 5.0]))
    rng = np.random.default_rng(0)
    mutated = mutate(ind, SPECS, mutation_rate=0.0, rng=rng)
    assert np.allclose(mutated.genes, ind.genes)


def test_mutate_does_not_modify_original():
    ind = Individual(genes=np.array([5.0, 5.0]))
    rng = np.random.default_rng(1)
    mutate(ind, SPECS, mutation_rate=1.0, rng=rng)
    assert np.allclose(ind.genes, [5.0, 5.0])  # original netaknut


def test_mutate_respects_bounds_even_with_full_rate():
    ind = Individual(genes=np.array([0.0, 10.0]))  # na samim granicama
    rng = np.random.default_rng(2)
    for _ in range(20):
        mutated = mutate(ind, SPECS, mutation_rate=1.0, rng=rng, sigma_fraction=0.5)
        for gene, spec in zip(mutated.genes, SPECS):
            assert spec.low <= gene <= spec.high


def test_mutate_invalid_rate_raises():
    ind = Individual(genes=np.array([1.0, 1.0]))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        mutate(ind, SPECS, mutation_rate=-0.1, rng=rng)
    with pytest.raises(ValueError):
        mutate(ind, SPECS, mutation_rate=1.1, rng=rng)