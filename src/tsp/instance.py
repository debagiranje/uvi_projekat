"""ucitavanje TSP instanci (TSPLIB format) i racunanje matrice udaljenosti"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class TSPInstance:
    """predstavlja jednu TSP instancu: koordinate gradova + matrica udaljenosti

    podrzava TSPLIB EDGE_WEIGHT_TYPE: EUC_2D (euklidsko rastojanje) i
    GEO (geografsko rastojanje po TSPLIB formuli)
    """

    def __init__(self, name: str, coords: np.ndarray, edge_weight_type: str = "EUC_2D"):
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords mora biti niz oblika (n_gradova, 2)")
        self.name = name
        self.coords = coords
        self.edge_weight_type = edge_weight_type
        self.n_cities = coords.shape[0]
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        if self.edge_weight_type == "GEO":
            return _geo_distance_matrix(self.coords)
        # default: euklidsko rastojanje (EUC_2D i slični tipovi)
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        dist = np.sqrt((diff**2).sum(axis=2))
        return dist

    def tour_length(self, tour: list[int] | np.ndarray) -> float:
        """racuna duzinu zatvorene rute (vraca se na polazni grad)"""
        tour = list(tour)
        if len(tour) != self.n_cities or set(tour) != set(range(self.n_cities)):
            raise ValueError("tour mora biti permutacija svih gradova instance")
        length = 0.0
        for i in range(len(tour)):
            a, b = tour[i], tour[(i + 1) % len(tour)]
            length += self.distance_matrix[a, b]
        return length

    @classmethod
    def from_tsplib_file(cls, path: str | Path) -> "TSPInstance":
        """loada .tsp fajl u TSPLIB formatu (NODE_COORD_SECTION)."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        name = path.stem
        edge_weight_type = "EUC_2D"
        coords: list[tuple[float, float]] = []
        in_coord_section = False

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("NAME"):
                name = stripped.split(":", 1)[-1].strip() or name
            elif stripped.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = stripped.split(":", 1)[-1].strip()
            elif stripped.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue
            elif stripped.startswith("EOF") or stripped.startswith("DISPLAY_DATA_SECTION"):
                break
            elif in_coord_section and stripped:
                parts = stripped.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))

        if not coords:
            raise ValueError(f"nije prondajen NODE_COORD_SECTION u {path}")

        return cls(name=name, coords=np.array(coords), edge_weight_type=edge_weight_type)

    @classmethod
    def random_instance(
        cls, n_cities: int, seed: int | None = None, size: float = 100.0
    ) -> "TSPInstance":
        """ generise random instancu za brzinske testove"""
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, size, size=(n_cities, 2))
        return cls(name=f"random_{n_cities}", coords=coords, edge_weight_type="EUC_2D")


def _geo_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """TSPLIB GEO dist: koordinate su u formatu DDD.MM (stepeni.minuti)
    konvertuju se u radijane pa se koristi great circle formula
    """
    PI = 3.141592
    RRR = 6378.388

    def to_radians(coord: np.ndarray) -> np.ndarray:
        deg = np.trunc(coord)
        minutes = coord - deg
        return PI * (deg + 5.0 * minutes / 3.0) / 180.0

    lat = to_radians(coords[:, 0])
    lon = to_radians(coords[:, 1])

    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        q1 = np.cos(lon[i] - lon)
        q2 = np.cos(lat[i] - lat)
        q3 = np.cos(lat[i] + lat)
        row = RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
        dist[i] = np.where(np.arange(n) == i, 0.0, row)
    return dist
