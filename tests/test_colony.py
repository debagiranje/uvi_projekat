from itertools import permutations

import pytest

from src.aco.colony import run_aco
from src.tsp.instance import TSPInstance


def test_run_aco_returns_valid_tour():
    inst = TSPInstance.random_instance(n_cities=12, seed=1)
    result = run_aco(
        inst.distance_matrix,
        n_ants=10,
        n_iterations=5,
        alpha=1.0,
        beta=2.0,
        rho=0.3,
        q=100.0,
        seed=42,
    )
    assert set(result.best_tour) == set(range(12))
    assert result.best_length > 0
    assert result.best_length == pytest.approx(inst.tour_length(result.best_tour))


def test_run_aco_convergence_is_monotonically_non_increasing():
    # best_length po iteraciji ne smije nikad porasti (uvijek pamtimo najbolju rutu do sad)
    inst = TSPInstance.random_instance(n_cities=15, seed=2)
    result = run_aco(
        inst.distance_matrix,
        n_ants=8,
        n_iterations=10,
        alpha=1.0,
        beta=2.0,
        rho=0.2,
        q=100.0,
        seed=7,
    )
    assert len(result.convergence) == 10
    for i in range(1, len(result.convergence)):
        assert result.convergence[i] <= result.convergence[i - 1]


def test_run_aco_deterministic_with_seed():
    inst = TSPInstance.random_instance(n_cities=10, seed=3)
    kwargs = dict(n_ants=6, n_iterations=5, alpha=1.0, beta=2.0, rho=0.3, q=100.0, seed=99)

    result_a = run_aco(inst.distance_matrix, **kwargs)
    result_b = run_aco(inst.distance_matrix, **kwargs)

    assert result_a.best_tour == result_b.best_tour
    assert result_a.best_length == result_b.best_length
    assert result_a.convergence == result_b.convergence


def test_run_aco_invalid_params_raise():
    inst = TSPInstance.random_instance(n_cities=5, seed=1)
    with pytest.raises(ValueError):
        run_aco(inst.distance_matrix, n_ants=0, n_iterations=5, alpha=1, beta=2, rho=0.3, q=1)
    with pytest.raises(ValueError):
        run_aco(inst.distance_matrix, n_ants=5, n_iterations=0, alpha=1, beta=2, rho=0.3, q=1)


def test_run_aco_finds_known_optimum_on_tiny_instance():
    # Regression test: na vrlo maloj instanci (5 gradova) sa dovoljno mrava/iteracija,
    # ACO mora naci globalni optimum (provjeren brute-force pretragom svih permutacija).
    inst = TSPInstance.random_instance(n_cities=5, seed=11)

    # brute-force optimum
    best_brute_force = min(
        inst.tour_length(list(perm)) for perm in permutations(range(5))
    )

    result = run_aco(
        inst.distance_matrix,
        n_ants=20,
        n_iterations=30,
        alpha=1.0,
        beta=3.0,
        rho=0.3,
        q=100.0,
        seed=42,
    )

    assert result.best_length == pytest.approx(best_brute_force, rel=1e-6)
