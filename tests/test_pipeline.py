from pathlib import Path

import pytest

from src.pipeline import build_param_specs, load_config, make_fitness_fn, run
from src.tsp.instance import TSPInstance

SMOKE_CONFIG = Path(__file__).parent.parent / "configs" / "smoke_test.yaml"


def test_load_config_reads_yaml():
    config = load_config(SMOKE_CONFIG)
    assert config["seed"] == 42
    assert "aco" in config
    assert "gp" in config


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/blablabla_ne_postoji.yaml")


def test_build_param_specs_extracts_range_keys():
    aco_config = {
        "n_ants": 20,
        "n_iterations": 100,
        "alpha_range": [0.5, 5.0],
        "beta_range": [0.5, 5.0],
    }
    specs = build_param_specs(aco_config)
    names = {s.name for s in specs}
    assert names == {"alpha", "beta"}


def test_build_param_specs_no_range_keys_raises():
    with pytest.raises(ValueError):
        build_param_specs({"n_ants": 5, "n_iterations": 3})


def test_make_fitness_fn_returns_positive_length():
    import numpy as np

    from src.gp.individual import Individual, ParameterSpec

    inst = TSPInstance.random_instance(n_cities=10, seed=1)
    specs = [
        ParameterSpec("alpha", 0.5, 5.0),
        ParameterSpec("beta", 0.5, 5.0),
        ParameterSpec("rho", 0.01, 0.5),
        ParameterSpec("q", 1.0, 500.0),
    ]
    rng = np.random.default_rng(0)
    fitness_fn = make_fitness_fn(
        inst.distance_matrix, specs, n_ants=5, n_iterations=3, aco_seed_rng=rng
    )

    individual = Individual(genes=np.array([1.0, 2.0, 0.3, 100.0]))
    fitness = fitness_fn(individual)

    assert fitness > 0


def test_run_end_to_end_with_smoke_config():
    config = load_config(SMOKE_CONFIG)
    result = run(config)

    assert result["status"] == "ok"
    assert result["best_tour_length"] > 0
    assert set(result["best_params"].keys()) == {"alpha", "beta", "rho", "q"}
    assert len(result["convergence"]) == config["gp"]["n_generations"]


def test_run_convergence_is_monotonically_non_increasing():
    config = load_config(SMOKE_CONFIG)
    result = run(config)
    history = result["convergence"]
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1]


def test_run_deterministic_with_same_seed():
    config = load_config(SMOKE_CONFIG)
    result_a = run(config)
    result_b = run(config)

    assert result_a["best_tour_length"] == result_b["best_tour_length"]
    assert result_a["best_params"] == result_b["best_params"]
    assert result_a["convergence"] == result_b["convergence"]

