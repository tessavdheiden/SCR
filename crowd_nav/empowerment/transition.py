import torch
import torch.nn as nn


class Transition(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Transition, self).__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(nb_states, hidden2)
        self.fc3 = nn.Linear(hidden1 + hidden2, nb_states)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, z):
        out1 = self.fc1(x)
        out1 = self.relu(out1)
        out2 = self.fc2(z)
        out2 = self.relu(out2)
        out = torch.cat([out1, out2], dim=1)
        out = self.fc3(out)
        out = self.tanh(out)
        return out
