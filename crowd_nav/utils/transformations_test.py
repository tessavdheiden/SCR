import unittest

import numpy as np
from crowd_nav.utils.transformations import build_occupancy_map, propagate
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.action import ActionXY


class BuildOccupancyMapTest(unittest.TestCase):
    def test_no_overlap(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.zeros((1, 4)) + 100
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2 * om_channel_size))
        self.assertTrue(np.array_equal(result, expected_result))

    def test_left_corner(self):
        pass

class PropagateTest(unittest.TestCase):
    def test_no_movement(self):
        radius = 1
        state = ObservableState(0, 0, 0, 0, radius)
        action = ActionXY(0, 0)
        next_state = propagate(state, action, time_step=.1, kinematics='holonomic')
        self.assertEqual(next_state, state)

if __name__ == "__main__":
    unittest.main()