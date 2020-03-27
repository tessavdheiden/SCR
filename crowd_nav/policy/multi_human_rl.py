import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.utils.transformations import build_occupancy_map


class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                       for human_state in state.human_states]
                    reward = self.compute_reward(next_self_state, next_human_states)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                              for next_human_state in next_human_states], dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states, next_self_state).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states, state.self_state)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states, robot_state):
        """
        :param human_states:
        :param robot_state
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = np.zeros((len(human_states), self.cell_num ** 2 * self.om_channel_size))
        for i, human in enumerate(human_states):
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states + [robot_state] if other_human != human], axis=0)
            dm = build_occupancy_map(human, other_humans, self.cell_num, self.cell_size, self.om_channel_size)
            occupancy_maps[i] = dm

        return torch.from_numpy(occupancy_maps).float()

    def make_patches(self, h, ax):
        import matplotlib.patches as patches
        self.locations = [(i % self.cell_num - self.cell_num // 2, i // self.cell_num - self.cell_num // 2) for i in
                          range(self.cell_num ** 2)]
        for i in range(self.cell_num ** 2):
            self.patches[h][i] = patches.Rectangle(self.locations[i], self.cell_size, self.cell_size, alpha=0.1)
            ax.add_artist(self.patches[h][i])

    def draw_observation(self, ax, ob, init=False):
        import matplotlib as mpl
        human_num = len(ob[1])
        if init:
            self.patches = [[None for _ in range(self.cell_num ** 2)] for _ in range(human_num)]
            for h in range(human_num):
                self.make_patches(h, ax)
        else:
            oms = self.build_occupancy_maps(ob[1], ob[0])
            for h in range(human_num):
                (px, py) = ob[1][h].position
                theta = np.arctan2(ob[1][h].vy, ob[1][h].vx)
                r = mpl.transforms.Affine2D().rotate(theta)
                t = mpl.transforms.Affine2D().translate(px, py)
                tra = r + t + ax.transData
                for c in range(self.cell_num ** 2):
                    self.patches[h][c].set_transform(tra)
                    self.patches[h][c].set_alpha(np.minimum(np.maximum(oms[h][c].item(), .2), .5))





