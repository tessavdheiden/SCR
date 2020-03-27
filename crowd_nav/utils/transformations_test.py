import unittest

from crowd_nav.utils.transformations import build_occupancy_map


class TransformationsTest(unittest.TestCase):
    def test_build_occupancy_map(self):
        human = []
        other_humans = []
        self.assertEqual(build_occupancy_map(human, other_humans), [])

if __name__ == "__main__":
    unittest.main()