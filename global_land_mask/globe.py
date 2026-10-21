"""Backward-compatible access to the original 1 km GLOBE land mask."""

from .mask import LandMask


_default = LandMask("globe-1km")


def lat_to_index(lat):
    """Convert latitude to an index on the legacy GLOBE mask."""
    lat_i, _ = _default._load().indices(lat, 0)
    return lat_i


def lon_to_index(lon):
    """Convert longitude to an index on the legacy GLOBE mask."""
    _, lon_i = _default._load().indices(0, lon)
    return lon_i


def is_land(lat, lon):
    """Return whether coordinates are on land using the 1 km GLOBE mask."""
    return _default.is_land(lat, lon)


def is_ocean(lat, lon):
    """Return whether coordinates are in the ocean using the 1 km GLOBE mask."""
    return _default.is_ocean(lat, lon)
