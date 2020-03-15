import torch
import torch.nn as nn
import torch.nn.functional as F


class Planning(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Planning, self).__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(nb_states, hidden1)
        self.fc = nn.Linear(hidden1 * 2, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden2)
        self.mu_head = nn.Linear(hidden2, nb_actions)
        self.sigma_head = nn.Linear(hidden2, nb_actions)

    def forward(self, s, s_):
        x = F.relu(self.fc1(s))
        x_ = F.relu(self.fc2(s_))
        x = F.relu(self.fc(torch.cat([x, x_], dim=-1)))
        x = F.relu(self.fc3(x))
        u = torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return u, sigma

