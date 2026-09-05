import numpy as np
import pytest

from src.gp.individual import (
    DEFAULT_ACO_PARAM_SPECS,
    Individual,
    ParameterSpec,
    clip_to_bounds,
    random_individual,
)


def test_parameter_spec_rejects_invalid_range():
    with pytest.raises(ValueError):
        ParameterSpec("bad", low=5.0, high=1.0)
    with pytest.raises(ValueError):
        ParameterSpec("bad", low=1.0, high=1.0)


def test_random_individual_respects_bounds():
    rng = np.random.default_rng(42)
    ind = random_individual(DEFAULT_ACO_PARAM_SPECS, rng)
    for gene, spec in zip(ind.genes, DEFAULT_ACO_PARAM_SPECS):
        assert spec.low <= gene <= spec.high


def test_random_individual_deterministic_with_seed():
    specs = DEFAULT_ACO_PARAM_SPECS
    ind_a = random_individual(specs, np.random.default_rng(7))
    ind_b = random_individual(specs, np.random.default_rng(7))
    assert np.allclose(ind_a.genes, ind_b.genes)


def test_as_dict_maps_genes_to_names():
    specs = [ParameterSpec("alpha", 0, 1), ParameterSpec("beta", 0, 1)]
    ind = Individual(genes=np.array([0.3, 0.7]))
    result = ind.as_dict(specs)
    assert result == {"alpha": pytest.approx(0.3), "beta": pytest.approx(0.7)}


def test_as_dict_mismatched_length_raises():
    specs = [ParameterSpec("alpha", 0, 1)]
    ind = Individual(genes=np.array([0.3, 0.7]))
    with pytest.raises(ValueError):
        ind.as_dict(specs)


def test_copy_is_independent():
    ind = Individual(genes=np.array([1.0, 2.0]), fitness=5.0)
    copy = ind.copy()
    copy.genes[0] = 999.0
    assert ind.genes[0] == pytest.approx(1.0)  # original nepromijenjen
    assert copy.fitness == ind.fitness


def test_clip_to_bounds_clamps_out_of_range_genes():
    specs = [ParameterSpec("a", 0.0, 1.0), ParameterSpec("b", 0.0, 10.0)]
    genes = np.array([-5.0, 999.0])
    clipped = clip_to_bounds(genes, specs)
    assert clipped[0] == pytest.approx(0.0)
    assert clipped[1] == pytest.approx(10.0)


def test_clip_to_bounds_mismatched_length_raises():
    specs = [ParameterSpec("a", 0.0, 1.0)]
    with pytest.raises(ValueError):
        clip_to_bounds(np.array([0.5, 0.5]), specs)