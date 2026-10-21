"""Fast global land/ocean classification."""

from .mask import LandMask
from . import globe
from .globe import is_land, is_ocean, lat_to_index, lon_to_index

__all__ = [
    "LandMask",
    "globe",
    "is_land",
    "is_ocean",
    "lat_to_index",
    "lon_to_index",
]
__version__ = "1.1.0"
