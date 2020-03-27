import numpy as np

def build_occupancy_map(human, other_humans):
    """
    :param human_states:
    :param robot_state
    :return: tensor of shape (# human - 1, self.cell_num ** 2)
    """
    other_px = other_humans[:, 0] - human.px
    other_py = other_humans[:, 1] - human.py
    # new x-axis is in the direction of human's velocity
    human_velocity_angle = np.arctan2(human.vy, human.vx)
    other_human_orientation = np.arctan2(other_py, other_px)
    rotation = other_human_orientation - human_velocity_angle
    distance = np.linalg.norm([other_px, other_py], axis=0)
    other_px = np.cos(rotation) * distance
    other_py = np.sin(rotation) * distance

    # compute indices of humans in the grid
    other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
    other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
    other_x_index[other_x_index < 0] = float('-inf')
    other_x_index[other_x_index >= self.cell_num] = float('-inf')
    other_y_index[other_y_index < 0] = float('-inf')
    other_y_index[other_y_index >= self.cell_num] = float('-inf')
    grid_indices = self.cell_num * other_y_index + other_x_index
    occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
    if self.om_channel_size == 1:
        return occupancy_map.astype(int)
    else:
        # calculate relative velocity for other agents
        other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
        rotation = other_human_velocity_angles - human_velocity_angle
        speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
        other_vx = np.cos(rotation) * speed
        other_vy = np.sin(rotation) * speed
        dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
        for i, index in np.ndenumerate(grid_indices):
            if index in range(self.cell_num ** 2):
                if self.om_channel_size == 2:
                    dm[2 * int(index)].append(other_vx[i])
                    dm[2 * int(index) + 1].append(other_vy[i])
                elif self.om_channel_size == 3:
                    dm[int(index)].append(1)
                    dm[int(index) + self.cell_num ** 2].append(other_vx[i])
                    dm[int(index) + self.cell_num ** 2 * 2].append(other_vy[i])
                else:
                    raise NotImplementedError
        for i, cell in enumerate(dm):
            dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
        return [dm]
