import logging
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, policy, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.policy = policy
        self.device = device
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size

    def set_learning_rate(self, lr):
        self.policy.set_device(self.device)
        self.policy.set_learning_rate(lr)

    def optimize_epoch(self, num_epochs):
        if self.policy.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                loss = self.policy.update(data)
                epoch_loss += loss

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.policy.optimizer is None: raise ValueError('Learning rate is not set!')

        if len(self.memory) < self.batch_size: raise ValueError('Not enough experiences collected!')

        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size)
        losses = 0
        for _ in range(num_batches):
            data = next(iter(self.data_loader))
            loss = self.policy.update(data)
            losses += loss

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
