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


class EmpowermentNetwork(nn.Module):
    def __init__(self, state_nb):
        super().__init__()
        self.source = Source(state_nb, 2)
        self.planning = Planning(state_nb, 2)
        self.transition = Transition(state_nb, 2)
        self.optimizer = optim.SGD(list(self.source.parameters()) + list(self.transition.parameters()) +
                                   list(self.planning.parameters()), lr=1e-4, momentum=0.9)

    def forward(self, human_state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        mu, sigma = self.source(human_state)
        dist_source = Normal(mu, sigma)
        sample = dist_source.rsample()
        human_next_state = self.transition.select_state(sample, human_state)
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
        self.with_om = config.getboolean('scr', 'with_om')
        if not self.with_om:
            raise AttributeError('SCR needs occupancy maps!')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')

        self.occupancy_map_dim = self.cell_num ** 2 * self.om_channel_size
        self.empowerment = EmpowermentNetwork(state_nb=self.occupancy_map_dim)

        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_occupancy_maps(self, joint_state):
        return joint_state[:, :, -self.occupancy_map_dim:]

    def update(self, data):
        inputs, values, _ = data
        inputs = Variable(inputs)
        values = Variable(values)
        human_states = Variable(self.get_occupancy_maps(inputs))

        self.empowerment.optimizer.zero_grad()
        estimate = -self.empowerment(human_states).mean()
        estimate.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(list(self.empowerment.planning.parameters()) + list(self.empowerment.source.parameters())+ list(self.empowerment.transition.parameters()),
                                 self.max_grad_norm)
        self.empowerment.optimizer.step()

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, values) + estimate.mean()
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

