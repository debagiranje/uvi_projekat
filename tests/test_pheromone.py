import numpy as np
import pytest

from src.aco.pheromone import PheromoneMatrix


def test_initial_matrix_shape_and_diagonal():
    pm = PheromoneMatrix(n_cities=5, initial_value=2.0)
    assert pm.tau.shape == (5, 5)
    assert np.allclose(np.diag(pm.tau), 0.0)
    # van-dijagonalni elementi = initial_value
    off_diag = pm.tau[~np.eye(5, dtype=bool)]
    assert np.allclose(off_diag, 2.0)


def test_initial_value_negative_raises():
    with pytest.raises(ValueError):
        PheromoneMatrix(n_cities=3, initial_value=-1.0)


def test_initial_value_zero_is_allowed():
    pm = PheromoneMatrix(n_cities=3, initial_value=0.0)
    assert np.allclose(pm.tau, 0.0)


def test_evaporate_reduces_pheromone_proportionally():
    pm = PheromoneMatrix(n_cities=4, initial_value=1.0)
    pm.evaporate(rho=0.5)
    off_diag = pm.tau[~np.eye(4, dtype=bool)]
    assert np.allclose(off_diag, 0.5)
    # dijagonala mora ostati 0 nakon isparavanja
    assert np.allclose(np.diag(pm.tau), 0.0)


def test_evaporate_invalid_rho_raises():
    pm = PheromoneMatrix(n_cities=3)
    with pytest.raises(ValueError):
        pm.evaporate(rho=0.0)
    with pytest.raises(ValueError):
        pm.evaporate(rho=1.0)
    with pytest.raises(ValueError):
        pm.evaporate(rho=1.5)


def test_deposit_adds_symmetrically():
    pm = PheromoneMatrix(n_cities=4, initial_value=0.0)
    pm.deposit(tour=[0, 1, 2, 3], amount=5.0)
    # ivica (0,1) i (1,0) moraju dobiti isti depozit
    assert pm.tau[0, 1] == pytest.approx(5.0)
    assert pm.tau[1, 0] == pytest.approx(5.0)
    # ivica (3,0) - zatvara rutu (posljednji -> prvi grad)
    assert pm.tau[3, 0] == pytest.approx(5.0)
    # ivica koja nije na ruti (0,2) mora ostati 0
    assert pm.tau[0, 2] == pytest.approx(0.0)


def test_deposit_from_ants_shorter_tour_gets_more_pheromone():
    pm = PheromoneMatrix(n_cities=4, initial_value=0.0)
    tours = [[0, 1, 2, 3], [0, 2, 1, 3]]
    lengths = [10.0, 20.0]  # prva ruta je duplo kracas
    pm.deposit_from_ants(tours, lengths, q=100.0)
    # ivica (0,1) postoji samo u prvoj ruti -> depozit = 100/10 = 10
    assert pm.tau[0, 1] == pytest.approx(10.0)
    # ivica (0,2) postoji samo u drugoj ruti -> depozit = 100/20 = 5
    assert pm.tau[0, 2] == pytest.approx(5.0)


def test_deposit_from_ants_nonpositive_length_raises():
    pm = PheromoneMatrix(n_cities=3, initial_value=0.0)
    with pytest.raises(ValueError):
        pm.deposit_from_ants(tours=[[0, 1, 2]], tour_lengths=[0.0], q=1.0)
