from torch.utils.data import Dataset
from collections import namedtuple


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
        self.experience = namedtuple("Experience", field_names=["state", "value", "human_state"])

    def push(self, item):
        # replace old experience with new experience
        e = self.experience(item[0], item[1], item[2])
        if len(self.memory) < self.position + 1:
            self.memory.append(e)
        else:
            self.memory[self.position] = e
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()
