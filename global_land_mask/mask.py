"""Configurable land-mask datasets."""

from pathlib import Path

import numpy as np


_DATA_DIR = Path(__file__).with_name("data")
_SUPPORTED = {
    ("globe", "1km"): "raster",
    ("natural-earth", "10m"): "polygon",
    ("natural-earth", "50m"): "polygon",
}


def _coordinates(lat, lon):
    lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    if np.any(~np.isfinite(lat)) or np.any(~np.isfinite(lon)):
        raise ValueError("latitude and longitude must be finite")
    if np.any((lat < -90) | (lat > 90)):
        raise ValueError("latitude must be between -90 and 90")
    if np.any((lon < -180) | (lon > 180)):
        raise ValueError("longitude must be between -180 and 180")
    return lat, lon


class _GlobeRaster:
    def __init__(self):
        data = np.load(Path(__file__).with_name("globe_combined_mask_compressed.npz"))
        self.mask = data["mask"]
        self.lat = data["lat"]
        self.lon = data["lon"]

    def indices(self, lat, lon):
        lat, lon = _coordinates(lat, lon)
        lat = np.clip(lat, self.lat.min(), self.lat.max())
        lon = np.clip(lon, self.lon.min(), self.lon.max())
        lat_i = ((lat - self.lat[0]) / (self.lat[1] - self.lat[0])).astype(int)
        lon_i = ((lon - self.lon[0]) / (self.lon[1] - self.lon[0])).astype(int)
        return lat_i, lon_i

    def is_land(self, lat, lon):
        lat_i, lon_i = self.indices(lat, lon)
        return np.logical_not(self.mask[lat_i, lon_i])


def _point_in_ring(x, y, ring):
    """Return True for points inside or on the boundary of a closed ring."""
    x1 = ring[:, 0]
    y1 = ring[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)

    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    scale = np.maximum.reduce((np.ones_like(x1), np.abs(x2 - x1), np.abs(y2 - y1)))
    on_line = np.abs(cross) <= 1e-12 * scale
    in_segment = (
        (np.minimum(x1, x2) - 1e-12 <= x)
        & (x <= np.maximum(x1, x2) + 1e-12)
        & (np.minimum(y1, y2) - 1e-12 <= y)
        & (y <= np.maximum(y1, y2) + 1e-12)
    )
    if np.any(on_line & in_segment):
        return True

    crosses = (y1 > y) != (y2 > y)
    x_cross = (x2[crosses] - x1[crosses]) * (y - y1[crosses]) / (
        y2[crosses] - y1[crosses]
    ) + x1[crosses]
    return bool(np.count_nonzero(x < x_cross) % 2)


class _NaturalEarthPolygons:
    def __init__(self, resolution, include_lakes_as_land):
        filename = _DATA_DIR / f"natural-earth-{resolution}-land.npz"
        data = np.load(filename)
        self.points = data["points"]
        self.offsets = data["ring_offsets"]
        self.bounds = data["bounds"]
        self.holes = data["is_hole"]
        self.include_lakes_as_land = include_lakes_as_land

    def _ring(self, index):
        return self.points[self.offsets[index] : self.offsets[index + 1]]

    def _one(self, lat, lon):
        candidates = np.flatnonzero(
            (self.bounds[:, 0] <= lon)
            & (lon <= self.bounds[:, 2])
            & (self.bounds[:, 1] <= lat)
            & (lat <= self.bounds[:, 3])
        )
        exterior = False
        in_hole = False
        for index in candidates:
            if _point_in_ring(lon, lat, self._ring(index)):
                if self.holes[index]:
                    in_hole = True
                else:
                    exterior = True
        return exterior and (self.include_lakes_as_land or not in_hole)

    def is_land(self, lat, lon):
        lat, lon = _coordinates(lat, lon)
        result = np.empty(lat.shape, dtype=bool)
        flat_result = result.reshape(-1)
        for index, (one_lat, one_lon) in enumerate(zip(lat.flat, lon.flat)):
            flat_result[index] = self._one(one_lat, one_lon)
        return result[()] if result.ndim == 0 else result


class LandMask:
    """Classify coordinates using a selectable global land-mask dataset.

    ``10m`` and ``50m`` are Natural Earth map-scale names (1:10 million and
    1:50 million), not distances in metres. Natural Earth lookups use polygon
    containment; the legacy GLOBE dataset uses its 1 km raster grid.
    """

    def __init__(
        self,
        dataset="globe",
        resolution=None,
        *,
        include_lakes_as_land=True,
    ):
        if dataset in {"globe-1km", "natural-earth-10m", "natural-earth-50m"}:
            dataset, resolution = dataset.rsplit("-", 1)
        dataset = str(dataset).lower().replace("_", "-")
        if dataset in {"naturalearth", "natural-earth-data"}:
            dataset = "natural-earth"
        if resolution is None:
            resolution = "1km" if dataset == "globe" else "10m"
        resolution = str(resolution).lower()
        if resolution.startswith("1:"):
            resolution = resolution[2:]
        key = (dataset, resolution)
        if key not in _SUPPORTED:
            choices = ", ".join(f"{d}-{r}" for d, r in _SUPPORTED)
            raise ValueError(f"unsupported land mask {dataset}-{resolution}; choose from {choices}")

        self.dataset = dataset
        self.resolution = resolution
        self.lookup = _SUPPORTED[key]
        self.include_lakes_as_land = bool(include_lakes_as_land)
        self._backend = None

    def _load(self):
        if self._backend is None:
            if self.lookup == "raster":
                self._backend = _GlobeRaster()
            else:
                self._backend = _NaturalEarthPolygons(
                    self.resolution, self.include_lakes_as_land
                )
        return self._backend

    def is_land(self, lat, lon):
        """Return whether each latitude/longitude coordinate is on land."""
        return self._load().is_land(lat, lon)

    def is_ocean(self, lat, lon):
        """Return whether each latitude/longitude coordinate is in the ocean."""
        return np.logical_not(self.is_land(lat, lon))

    def __repr__(self):
        return (
            f"LandMask(dataset={self.dataset!r}, resolution={self.resolution!r}, "
            f"lookup={self.lookup!r})"
        )
