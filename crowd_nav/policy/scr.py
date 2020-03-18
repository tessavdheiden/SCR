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
    def __init__(self):
        super().__init__()
        self.source = Source(5, 2)
        self.planning = Planning(5, 2)
        self.transition = Transition(5, 2)
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
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        self.empowerment = EmpowermentNetwork()

        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def update(self, data):
        inputs, values, human_states = data
        inputs = Variable(inputs)
        values = Variable(values)
        human_states = Variable(human_states)

        self.empowerment.optimizer.zero_grad()
        estimate = self.empowerment(human_states).mean()
        estimate.backward(retain_graph=True)
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
            human_states = torch.cat([torch.Tensor([human_state.px, human_state.py, human_state.vx, human_state.vy,
                                                    human_state.radius]).to(self.device)
                                      for human_state in ob[1]], dim=0).reshape(1, -1, 5)
            empowerment = self.empowerment(human_states)
            for h in range(human_num):
                self.scores[h].set_text('human {}: {:.2f}'.format(h, empowerment[0, h].mean()))

