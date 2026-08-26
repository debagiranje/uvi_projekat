"""
main entry point koji spaja GP i ACO slojeve
za sada samo config in, res out koji CI može provjeriti
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """ucitava YAML konfiguraciju eksperimenta"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config file {path} mora biti dict")
    return config


def run(config: dict) -> dict:
    """
    run placeholder
    """
    return {
        "seed": config.get("seed"),
        "instance": config.get("instance"),
        "best_tour_length": None,
        "status": "skeleton, TBD",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ACO + GP za TSP")
    parser.add_argument("--config", required=True, help="putanja do YAML config filea")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run(config)
    print(result)


if __name__ == "__main__":
    main()
