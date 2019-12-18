import numpy as np
import torch
import torch.nn as nn

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Planning(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Planning, self).__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(nb_states, hidden2)
        self.fc3 = nn.Linear(hidden1 + hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, z, z_next):
        out1 = self.fc1(z)
        out1 = self.relu(out1)
        out2 = self.fc2(z_next)
        out2 = self.relu(out2)
        out = torch.cat([out1, out2], dim=1)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

    def select_action(self, z, z_next):
        batch_size, _, _ = z.shape
        z = z.view(-1, self.nb_states)
        z_next = z_next.view(-1, self.nb_states)
        out = self.forward(z, z_next)
        return out.view(batch_size, -1, self.nb_actions)
