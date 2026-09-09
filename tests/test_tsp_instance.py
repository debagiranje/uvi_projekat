import numpy as np
import pytest

from src.tsp.instance import TSPInstance


def test_random_instance_shape_and_symmetry():
    inst = TSPInstance.random_instance(n_cities=10, seed=42)
    assert inst.n_cities == 10
    assert inst.distance_matrix.shape == (10, 10)
    # matrica udaljenosti mora biti simetrična (neusmjeren graf)
    assert np.allclose(inst.distance_matrix, inst.distance_matrix.T)
    # dijagonala mora biti 0 (rastojanje grada od samog sebe)
    assert np.allclose(np.diag(inst.distance_matrix), 0.0)


def test_random_instance_deterministic_with_seed():
    a = TSPInstance.random_instance(n_cities=5, seed=123)
    b = TSPInstance.random_instance(n_cities=5, seed=123)
    assert np.allclose(a.coords, b.coords)
    assert np.allclose(a.distance_matrix, b.distance_matrix)


def test_tour_length_simple_square():
    # 4 grada u kvadratu stranice 1 - poznata duzina obilaska = 4.0
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    inst = TSPInstance(name="square", coords=coords, edge_weight_type="EUC_2D")
    length = inst.tour_length([0, 1, 2, 3])
    assert length == pytest.approx(4.0)


def test_tour_length_invalid_tour_raises():
    coords = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    inst = TSPInstance(name="triangle", coords=coords)
    with pytest.raises(ValueError):
        inst.tour_length([0, 1])  # nedostaje grad
    with pytest.raises(ValueError):
        inst.tour_length([0, 1, 1])  # ponovljen grad


def test_from_tsplib_file_loads_burma14(tmp_path):
    # burma proba
    from pathlib import Path

    burma_path = Path(__file__).parent.parent / "data" / "instances" / "burma14.tsp"
    inst = TSPInstance.from_tsplib_file(burma_path)
    assert inst.name == "burma14"
    assert inst.n_cities == 14
    assert inst.edge_weight_type == "GEO"
    assert inst.distance_matrix.shape == (14, 14)
    assert np.allclose(np.diag(inst.distance_matrix), 0.0)
