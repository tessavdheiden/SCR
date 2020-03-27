import unittest

import numpy as np
from crowd_nav.utils.transformations import build_occupancy_map
from crowd_sim.envs.utils.human import Human


class TransformationsTest(unittest.TestCase):
    def test_build_occupancy_map(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_humans = np.zeros((1, 4))
        self.assertEqual(build_occupancy_map(human, other_humans, cell_num, cell_size, om_channel_size), np.zeros((cell_num ** 2 * om_channel_size)))

if __name__ == "__main__":
    unittest.main()