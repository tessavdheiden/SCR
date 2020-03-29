import unittest

import numpy as np
from crowd_nav.utils.transformations import build_occupancy_map, propagate, propagate_occupancy_map, get_states_from_occupancy_map
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.action import ActionXY, ActionRot


class BuildOccupancyMapTest(unittest.TestCase):
    def test_no_overlap(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.zeros((1, 4)) + 100
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2))
        self.assertTrue(np.array_equal(result, expected_result))

    def test_left_upper_corner(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.array([-1.5, 1.5, 0, 0]).reshape(1, -1)
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2))
        expected_result[12] = 1
        self.assertTrue(np.array_equal(result, expected_result))

    def test_right_lower_corner(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.array([1.5, -1.5, 0, 0]).reshape(1, -1)
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2))
        expected_result[3] = 1
        self.assertTrue(np.array_equal(result, expected_result))

    def test_right_upper_corner(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.array([1.5, 1.5, 0, 0]).reshape(1, -1)
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2))
        expected_result[15] = 1
        self.assertTrue(np.array_equal(result, expected_result))

    def test_human_rotated_counter_clockwise(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=np.sqrt(2), vy=np.sqrt(2), gx=0, gy=0, theta=0)
        other_agents = np.array([1.5, 0, 0, 0]).reshape(1, -1)
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2))
        expected_result[3] = 1
        self.assertTrue(np.array_equal(result, expected_result))

    def test_human_rotated_clockwise(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=np.sqrt(2), vy=-np.sqrt(2), gx=0, gy=0, theta=0)
        other_agents = np.array([1.5, 0, 0, 0]).reshape(1, -1)
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2))
        expected_result[15] = 1
        self.assertTrue(np.array_equal(result, expected_result))

    def test_human_map_three_channels(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 3
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.array([1.5, 1.5, 1, 2]).reshape(1, -1)
        result = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
        expected_result = np.zeros((cell_num ** 2 * om_channel_size))
        expected_result[15] = 1
        expected_result[15 + cell_num ** 2] = 1
        expected_result[15 + cell_num ** 2 * 2] = 2
        self.assertTrue(np.allclose(result, expected_result, atol=1e-5))

class PropagateTest(unittest.TestCase):
    def test_no_movement(self):
        radius = 1
        state = ObservableState(0, 0, 0, 0, radius)
        action = ActionXY(0, 0)
        next_state = propagate(state, action, time_step=.1, kinematics='holonomic')
        self.assertEqual(next_state, state)

    def test_holonomic_diagonal_up_movement(self):
        radius = 1
        time_step = .1
        state = ObservableState(0, 0, 0, 0, radius)
        action = ActionXY(np.sqrt(2), np.sqrt(2))
        next_state = propagate(state, action, time_step=time_step, kinematics='holonomic')
        expected_state = ObservableState(time_step * np.sqrt(2), time_step * np.sqrt(2), action.vx, action.vy, radius)
        self.assertEqual(next_state, expected_state)

    def test_non_holonomic_left_movement(self):
        radius = 1
        time_step = .1
        state = FullState(0, 0, 0, 0, radius, 0, 0, 0, 0)
        r = np.pi
        action = ActionRot(1., r)
        next_state = propagate(state, action, time_step=time_step, kinematics='non_holonomic')
        expected_state = FullState(time_step * np.cos(r), time_step * np.sin(r), \
                                         np.cos(r), np.sin(r), radius, 0, 0, 0, r)

        self.assertEqual(expected_state, next_state)


class PropagateOccupancyMapTest(unittest.TestCase):
    def test_no_overlap_no_movement(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.zeros((1, 4)) + 100
        occupancy_map = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)

        radius = 1
        state = ObservableState(0, 0, 0, 0, radius)
        action = ActionXY(0, 0)
        next_state = propagate(state, action, time_step=.1, kinematics='holonomic')

        next_occupancy_map = propagate_occupancy_map(occupancy_map, action, time_step=.1, kinematics='holonomic')
        expected_next_occupancy_map = build_occupancy_map(next_state, other_agents, cell_num, cell_size, om_channel_size)

        self.assertTrue(np.array_equal(next_occupancy_map, expected_next_occupancy_map))

    def test_overlap_no_movement(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.zeros((1, 4))
        occupancy_map = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)

        radius = 1
        state = ObservableState(0, 0, 0, 0, radius)
        action = ActionXY(0, 0)
        next_state = propagate(state, action, time_step=.1, kinematics='holonomic')

        next_occupancy_map = propagate_occupancy_map(occupancy_map, action, time_step=.1, kinematics='holonomic')
        expected_next_occupancy_map = build_occupancy_map(next_state, other_agents, cell_num, cell_size,
                                                          om_channel_size)

        self.assertTrue(np.array_equal(next_occupancy_map, expected_next_occupancy_map))

    def test_states_from_om_one_other_agent(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 1
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.zeros((1, 4))
        for i in np.linspace(-2, 2, 9):
            for j in np.linspace(-2, 2, 9):
                other_agents[0, 0] = i
                other_agents[0, 1] = j
                occupancy_map = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
                states = get_states_from_occupancy_map(occupancy_map, cell_num, cell_size, om_channel_size)
                if states.shape[0]>0:
                    self.assertTrue(np.allclose(states, other_agents, atol=cell_size / 2.))

    def test_states_from_om_one_other_agent_with_three_channels(self):
        cell_num = 4
        cell_size = 1
        om_channel_size = 3
        human = Human()
        human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
        other_agents = np.zeros((1, 4))
        for i in np.linspace(-2, 2, 9):
            for j in np.linspace(-2, 2, 9):
                other_agents[0, 0] = i
                other_agents[0, 1] = j
                other_agents[0, 2] = np.random.rand(1)
                other_agents[0, 3] = np.random.rand(1)
                occupancy_map = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
                states = get_states_from_occupancy_map(occupancy_map, cell_num, cell_size, om_channel_size)
                if states.shape[0]>0:
                    self.assertTrue(np.allclose(states, other_agents, atol=cell_size / 2.))


# def test_overlap_movement(self):
    #     cell_num = 4
    #     cell_size = 1
    #     om_channel_size = 1
    #     human = Human()
    #     human.set(px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0)
    #     other_agents = np.zeros((1, 4))
    #     occupancy_map = build_occupancy_map(human, other_agents, cell_num, cell_size, om_channel_size)
    #
    #     radius = 1
    #     state = ObservableState(0, 0, 0, 0, radius)
    #     action = ActionXY(np.sqrt(2), np.sqrt(2))
    #     next_state = propagate(state, action, time_step=.1, kinematics='holonomic')
    #
    #     next_occupancy_map = propagate_occupancy_map(occupancy_map, action, time_step=.1, kinematics='holonomic')
    #     expected_next_occupancy_map = build_occupancy_map(next_state, other_agents, cell_num, cell_size,
    #                                                       om_channel_size)
    #
    #     self.assertTrue(np.array_equal(next_occupancy_map, expected_next_occupancy_map))


if __name__ == "__main__":
    unittest.main()