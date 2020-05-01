import logging
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
from torch.autograd import Variable

from crowd_nav.policy.sarl import SARL, ValueNetwork
from crowd_nav.empowerment.source import Source
from crowd_nav.empowerment.planning import Planning
from crowd_nav.empowerment.transition import Transition

from crowd_nav.utils.transformations import build_occupancy_map_torch


class EmpowermentNetwork(nn.Module):
    def __init__(self, state_nb):
        super().__init__()
        self.source = Source(state_nb, 2)
        self.planning = Planning(state_nb, 2)
        self.transition = Transition(state_nb, 2)
        self.params = list(self.source.parameters()) + list(self.planning.parameters())
        self.optimizer = optim.SGD(self.params, lr=1e-4, momentum=0.9)
        self.optimizer_transition = optim.SGD(self.transition.parameters(), lr=1e-4, momentum=0.9)

    def forward(self, human_state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        mu, sigma = self.source(human_state)
        dist_source = Normal(mu, sigma)
        sample = dist_source.rsample()
        human_next_state = self.transition(sample, human_state)
        mu_p, sigma_p = self.planning(human_state, human_next_state)
        dist_plan = Normal(mu_p, sigma_p)

        return dist_plan.log_prob(sample) - dist_source.log_prob(sample)


class SCR(SARL):
    max_grad_norm = .5
    def __init__(self):
        super().__init__()
        self.name = 'SCR'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        if not self.with_om: raise AttributeError('SCR needs occupancy maps!')
        with_global_state = config.getboolean('sarl', 'with_global_state')

        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        self.time_step = .25

        self.occupancy_map_dim = self.cell_num ** 2 * self.om_channel_size
        self.empowerment = EmpowermentNetwork(state_nb=self.occupancy_map_dim)
        self.beta = config.getfloat('scr', 'beta')
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_human_next_state(self, state):
        """
        Propagates a state of a human from its current state.
        :param state: tensor of shape (5, ) px, py, vx, vy, radius
        :return: tensor of shape (5, )
        """
        next_px = state[0] + state[2] * self.time_step
        next_py = state[1] + state[3] * self.time_step
        return torch.tensor([next_px, next_py, state[2], state[3], state[4]])

    def update(self, data):
        joint_states, values, states = data
        n_batch, n_humans, _ = joint_states.shape
        _, _, n_states = states.shape

        joint_states = Variable(joint_states)
        values = Variable(values)
        human_occupancy_maps = Variable(joint_states[:, :, -self.occupancy_map_dim:]).view(-1, self.occupancy_map_dim)

        # Compute empowerment and update source and planning
        self.empowerment.optimizer.zero_grad()
        estimate = self.empowerment.forward(human_occupancy_maps).mean(-1) # take mean over cells
        loss = - estimate.mean()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.empowerment.params, self.max_grad_norm)
        self.empowerment.optimizer.step()

        # Train transition, sanity check: print(human_actions[4, 3]), print(human_actions.reshape(-1, 2)[23])
        states = Variable(states)

        human_states = states[:, :-1]
        human_actions = human_states[:, :, 2:4]
        prediction = self.empowerment.transition(human_actions.reshape(-1, 2), human_occupancy_maps) # not add robot,
        human_oms_next = torch.zeros(n_batch, n_humans, self.occupancy_map_dim)

        # Propagate human states and compute occupancy maps
        for i, scene in enumerate(human_states):
            for j, human in enumerate(scene):
                # propagate human
                human_next = self.get_human_next_state(human)
                others = torch.zeros(n_humans, n_states) # including robot
                for k, other in enumerate(scene):
                    if k == j:
                        others[k] = states[i, -1] # put robot state at human's own state
                    else:
                        others[k] = other

                human_oms_next[i, j, :] = build_occupancy_map_torch(human_next, others, self.cell_num, self.cell_size, self.om_channel_size)

        self.empowerment.optimizer_transition.zero_grad()
        error = self.criterion(prediction, human_oms_next.view(-1, self.occupancy_map_dim))
        error.backward()
        nn.utils.clip_grad_norm_(self.empowerment.transition.parameters(), self.max_grad_norm)
        self.empowerment.optimizer_transition.step()

        # train value network wit empowerment
        self.optimizer.zero_grad()
        outputs = self.model(joint_states)
        loss = self.criterion(outputs, (1 - self.beta) * values + self.beta * estimate.view(n_batch, n_humans).mean(-1).view(-1, 1))
        loss.backward()
        self.optimizer.step()

        return loss.data.item()

    def draw_attention(self, ax, ob, init=False):
        human_num = len(ob[1])
        if init:
            self.scores = [None for _ in range(human_num)]
            for h in range(human_num):
                self.make_text(h, ax)
        else:
            oms = self.build_occupancy_maps(ob[1], ob[0]).unsqueeze(0)
            empowerment = self.empowerment(oms)
            for h in range(human_num):
                self.scores[h].set_text('human {}: {:.2f}'.format(h, empowerment[0, h].mean()))

