import numpy as np
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import State, ObservableState, FullState
from crowd_sim.envs.utils.action import Action


def build_occupancy_map(human : Human, other_agents : np.array, cell_num : int, cell_size : float, om_channel_size : int) -> np.array:
    """
    :param human_states:
    :param robot_state
    :return: tensor of shape (# human - 1, cell_num ** 2)
    """
    other_px = other_agents[:, 0] - human.px
    other_py = other_agents[:, 1] - human.py
    # new x-axis is in the direction of human's velocity
    human_velocity_angle = np.arctan2(human.vy, human.vx)
    other_human_orientation = np.arctan2(other_py, other_px)
    rotation = other_human_orientation - human_velocity_angle
    distance = np.linalg.norm([other_px, other_py], axis=0)
    other_px = np.cos(rotation) * distance
    other_py = np.sin(rotation) * distance

    # compute indices of humans in the grid
    other_x_index = np.floor(other_px / cell_size + cell_num / 2)
    other_y_index = np.floor(other_py / cell_size + cell_num / 2)
    other_x_index[other_x_index < 0] = float('-inf')
    other_x_index[other_x_index >= cell_num] = float('-inf')
    other_y_index[other_y_index < 0] = float('-inf')
    other_y_index[other_y_index >= cell_num] = float('-inf')
    grid_indices = cell_num * other_y_index + other_x_index
    occupancy_map = np.isin(range(cell_num ** 2), grid_indices)
    if om_channel_size == 1:
        return occupancy_map.astype(int)
    else:
        # calculate relative velocity for other agents
        other_human_velocity_angles = np.arctan2(other_agents[:, 3], other_agents[:, 2])
        rotation = other_human_velocity_angles - human_velocity_angle
        speed = np.linalg.norm(other_agents[:, 2:4], axis=1)
        other_vx = np.cos(rotation) * speed
        other_vy = np.sin(rotation) * speed
        dm = [list() for _ in range(cell_num ** 2 * om_channel_size)]
        for i, index in np.ndenumerate(grid_indices):
            if index in range(cell_num ** 2):
                if om_channel_size == 2:
                    dm[2 * int(index)].append(other_vx[i])
                    dm[2 * int(index) + 1].append(other_vy[i])
                elif om_channel_size == 3:
                    dm[int(index)].append(1)
                    dm[int(index) + cell_num ** 2].append(other_vx[i])
                    dm[int(index) + cell_num ** 2 * 2].append(other_vy[i])
                else:
                    raise NotImplementedError
        for i, cell in enumerate(dm):
            dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
        return dm


def propagate(state : State, action : Action, time_step : float, kinematics : str) -> State:
    if isinstance(state, ObservableState):
        # propagate state of humans
        next_px = state.px + action.vx * time_step
        next_py = state.py + action.vy * time_step
        next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
    elif isinstance(state, FullState):
        # propagate state of current agent
        # perform action without rotation
        if kinematics == 'holonomic':
            next_px = state.px + action.vx * time_step
            next_py = state.py + action.vy * time_step
            next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                   state.gx, state.gy, state.v_pref, state.theta)
        elif kinematics == 'unicycle':
            # altered for Turtlebot:
            next_theta = state.theta + (action.r * time_step)
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            if action.r == 0:
                next_px = state.px + action.v * np.cos(state.theta) * time_step
                next_py = state.py + action.v * np.sin(state.theta) * time_step
            else:
                next_px = state.px + (action.v / action.r) * (
                        np.sin(action.r * time_step + state.theta) - np.sin(state.theta))
                next_py = state.py + (action.v / action.r) * (
                        np.cos(state.theta) - np.cos(action.r * time_step + state.theta))
            next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                   state.v_pref, next_theta)
        else:
            next_theta = state.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            next_px = state.px + next_vx * time_step
            next_py = state.py + next_vy * time_step
            next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                   state.v_pref, next_theta)
    else:
        raise ValueError('Type error')

    return next_state


def get_states_from_occupancy_map(occupancy_map : np.array, cell_num : int, cell_size : float, om_channel_size : int):
    indeces = np.nonzero(occupancy_map[0:(cell_num ** 2)])
    states = np.zeros((indeces[0].shape[0], 4))
    if om_channel_size == 1:
        for i, idx in enumerate(indeces[0]):
            row = idx // cell_num - cell_num // 2
            col = idx % cell_num - cell_num // 2
            states[i] = np.array([col*cell_size + cell_size / 2, row*cell_size + cell_size / 2, 0, 0])
    else:
        for i, idx in enumerate(indeces[0]):
            row = idx // cell_num - cell_num // 2
            col = idx % cell_num - cell_num // 2
            px = col*cell_size + cell_size / 2
            py = row*cell_size + cell_size / 2
            vx = occupancy_map[idx + cell_num**2]
            vy = occupancy_map[idx + cell_num**2*2]
            states[i] = np.array([px, py, vx, vy])
    return states


def propagate_occupancy_map(occupancy_map, state, action, time_step, kinematics, cell_num, cell_size, om_channel_size) -> np.array:
    states = get_states_from_occupancy_map(occupancy_map, cell_num, cell_size, om_channel_size)
    next_state = propagate(state, action, time_step, kinematics)
    return build_occupancy_map(next_state, states, cell_num, cell_size, om_channel_size)
