import numpy as np
import pytest

from src.gp.evolve import run_gp
from src.gp.individual import Individual, ParameterSpec

# 2d opseg parametara za testiranje GP logike NEZAVISNO od ACE
SPECS = [ParameterSpec("x", 0.0, 10.0), ParameterSpec("y", 0.0, 10.0)]
TARGET = np.array([3.0, 7.0])


def sphere_fitness(individual: Individual) -> float:
    """Mock fitness funkcija: kvadrat udaljenosti od poznate tacke (3, 7).

    Minimum (fitness=0) je poznat unaprijed - ovo nam omogucava da
    provjerimo da GP stvarno konvergira ka optimumu, nezavisno od
    bilo kakve ACO/TSP logike
    """
    return float(np.sum((individual.genes - TARGET) ** 2))


def test_run_gp_returns_valid_result_structure():
    result = run_gp(
        sphere_fitness, SPECS,
        population_size=10, n_generations=5,
        crossover_rate=0.8, mutation_rate=0.2, tournament_size=3, seed=1,
    )
    assert len(result.best_fitness_history) == 5
    assert result.best_individual.fitness is not None
    assert len(result.best_individual.genes) == 2


def test_run_gp_convergence_is_monotonically_non_increasing():
    result = run_gp(
        sphere_fitness, SPECS,
        population_size=15, n_generations=20,
        crossover_rate=0.8, mutation_rate=0.2, tournament_size=3, seed=5,
    )
    history = result.best_fitness_history
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1]


def test_run_gp_converges_toward_known_optimum():
    # sa dovoljno generacija, GP treba znacajno da se priblizi poznatom
    # minimumu (fitness = 0 kod tacke (3,7)), pocevsi od nasumicne populacije.
    result = run_gp(
        sphere_fitness, SPECS,
        population_size=30, n_generations=50,
        crossover_rate=0.8, mutation_rate=0.15, tournament_size=3, seed=42,
    )
    assert result.best_individual.fitness < 0.5  # blizu 0 (poznat minimum)


def test_run_gp_deterministic_with_seed():
    kwargs = dict(
        fitness_fn=sphere_fitness, specs=SPECS,
        population_size=10, n_generations=8,
        crossover_rate=0.8, mutation_rate=0.2, tournament_size=3, seed=99,
    )
    result_a = run_gp(**kwargs)
    result_b = run_gp(**kwargs)
    assert np.allclose(result_a.best_individual.genes, result_b.best_individual.genes)
    assert result_a.best_fitness_history == result_b.best_fitness_history


def test_run_gp_elitism_never_loses_best_solution():
    # Cak i sa elitism=0 (nema direktnog kopiranja), best_fitness_history
    # nikad ne smije rasti - jer se best_individual prati odvojeno
    result = run_gp(
        sphere_fitness, SPECS,
        population_size=10, n_generations=15,
        crossover_rate=0.9, mutation_rate=0.3, tournament_size=2, seed=3,
        elitism=0,
    )
    history = result.best_fitness_history
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1]


def test_run_gp_invalid_params_raise():
    kwargs = dict(
        fitness_fn=sphere_fitness, specs=SPECS,
        population_size=10, n_generations=5,
        crossover_rate=0.8, mutation_rate=0.2, tournament_size=3, seed=1,
    )
    with pytest.raises(ValueError):
        run_gp(**{**kwargs, "population_size": 1})
    with pytest.raises(ValueError):
        run_gp(**{**kwargs, "n_generations": 0})
    with pytest.raises(ValueError):
        run_gp(**{**kwargs, "elitism": 10})
    with pytest.raises(ValueError):
        run_gp(**{**kwargs, "elitism": -1})