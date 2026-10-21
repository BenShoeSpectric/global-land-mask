import unittest

import numpy as np

from global_land_mask import LandMask, globe, is_land


class LandMaskTests(unittest.TestCase):
    def test_dataset_selection_and_aliases(self):
        detailed = LandMask("natural-earth-10m")
        medium = LandMask(dataset="naturalearth", resolution="1:50m")
        self.assertEqual((detailed.dataset, detailed.resolution), ("natural-earth", "10m"))
        self.assertEqual((medium.dataset, medium.resolution), ("natural-earth", "50m"))
        self.assertEqual(detailed.lookup, "polygon")

    def test_natural_earth_known_points(self):
        for resolution in ("10m", "50m"):
            mask = LandMask("natural-earth", resolution)
            self.assertTrue(mask.is_land(38.0, -97.0))  # continental United States
            self.assertTrue(mask.is_land(-33.8688, 151.2093))  # Sydney
            self.assertFalse(mask.is_land(0.0, -140.0))  # Pacific Ocean
            self.assertTrue(mask.is_ocean(0.0, -140.0))

    def test_scale_changes_coastline_detail(self):
        # Manhattan is represented by the 10m land geometry but omitted at 50m.
        self.assertTrue(LandMask("natural-earth-10m").is_land(40.7128, -74.0060))
        self.assertFalse(LandMask("natural-earth-50m").is_land(40.7128, -74.0060))

    def test_arrays_and_broadcasting(self):
        result = LandMask("natural-earth-10m").is_land(
            np.array([38.0, 0.0]), -97.0
        )
        np.testing.assert_array_equal(result, [True, False])

    def test_coordinate_validation(self):
        mask = LandMask("natural-earth-10m")
        with self.assertRaises(ValueError):
            mask.is_land(91, 0)
        with self.assertRaises(ValueError):
            mask.is_land(0, 181)
        with self.assertRaises(ValueError):
            mask.is_land(np.nan, 0)

    def test_unsupported_dataset_message(self):
        with self.assertRaisesRegex(ValueError, "choose from"):
            LandMask("natural-earth", "110m")

    def test_legacy_api(self):
        self.assertTrue(bool(globe.is_land(40, -120)))
        self.assertTrue(bool(globe.is_ocean(40, -150)))
        self.assertTrue(bool(is_land(40, -120)))


if __name__ == "__main__":
    unittest.main()
