import numpy as np
import pytest

from src.aco.construct import construct_tour
from src.tsp.instance import TSPInstance


def test_construct_tour_is_valid_permutation():
    inst = TSPInstance.random_instance(n_cities=15, seed=1)
    pheromone = np.ones((15, 15))
    rng = np.random.default_rng(7)

    tour = construct_tour(inst.distance_matrix, pheromone, alpha=1.0, beta=2.0, rng=rng)

    assert len(tour) == 15
    assert set(tour) == set(range(15))  # svaki grad posjecen tacno jednom


def test_construct_tour_respects_start_city():
    inst = TSPInstance.random_instance(n_cities=8, seed=2)
    pheromone = np.ones((8, 8))
    rng = np.random.default_rng(3)

    tour = construct_tour(
        inst.distance_matrix, pheromone, alpha=1.0, beta=2.0, rng=rng, start_city=4
    )
    assert tour[0] == 4


def test_construct_tour_deterministic_with_same_rng_state():
    inst = TSPInstance.random_instance(n_cities=10, seed=1)
    pheromone = np.ones((10, 10))

    tour_a = construct_tour(
        inst.distance_matrix, pheromone, alpha=1.0, beta=2.0,
        rng=np.random.default_rng(99), start_city=0,
    )
    tour_b = construct_tour(
        inst.distance_matrix, pheromone, alpha=1.0, beta=2.0,
        rng=np.random.default_rng(99), start_city=0,
    )
    assert tour_a == tour_b


def test_construct_tour_zero_pheromone_falls_back_to_uniform_random():
    # degenerisan slucaj: svi feromoni 0 - ne smije puci, mora vratiti validnu rutu
    inst = TSPInstance.random_instance(n_cities=6, seed=5)
    pheromone = np.zeros((6, 6))
    rng = np.random.default_rng(1)

    tour = construct_tour(inst.distance_matrix, pheromone, alpha=1.0, beta=2.0, rng=rng)
    assert set(tour) == set(range(6))


def test_construct_tour_prefers_shorter_edges_with_high_beta():
    # sa vrlo visokim beta (heuristika dominira) i alpha=0 (feromon ignorisan)
    # mrav treba skoro uvijek da bira najblizi sljedeci grad, pohlepno
    coords = np.array([[0, 0], [1, 0], [10, 0], [2, 0]], dtype=float)
    inst = TSPInstance(name="line", coords=coords, edge_weight_type="EUC_2D")
    pheromone = np.ones((4, 4))
    rng = np.random.default_rng(0)

    tour = construct_tour(
        inst.distance_matrix, pheromone, alpha=0.0, beta=10.0, rng=rng, start_city=0
    )
    # iz grada 0, najblizi je grad 1 (udaljenost 1), zatim grad 3 (udaljenost 2 od 1)
    assert tour[1] == 1


def test_construct_tour_shape_mismatch_raises():
    distance_matrix = np.ones((5, 5))
    pheromone = np.ones((4, 4))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        construct_tour(distance_matrix, pheromone, alpha=1.0, beta=1.0, rng=rng)
