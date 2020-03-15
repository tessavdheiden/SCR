import logging
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import Variable
from torch.utils.data import DataLoader
from crowd_nav.empowerment.source import Source
from crowd_nav.empowerment.planning import Planning
from crowd_nav.empowerment.transition import Transition


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.source = Source(5, 2)
        self.planning = Planning(5, 2)
        self.transition = Transition(5, 2)

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values, human_state = data
                inputs = Variable(inputs)
                values = Variable(values)
                human_state = Variable(human_state)

                # self.optimizer.step()
                self.optimizer.zero_grad()
                mu, sigma = self.source(human_state)
                dist_source = Normal(mu, sigma)
                sample = dist_source.rsample()
                human_next_state = self.transition.select_state(sample, human_state)
                mu_p, sigma_p = self.planning(human_state, human_next_state)
                dist_plan = Normal(mu_p, sigma_p)

                MI = (dist_plan.log_prob(sample) - dist_source.log_prob(sample)).mean()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, values) - MI
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values, human_state = next(iter(self.data_loader))
            inputs = Variable(inputs)
            values = Variable(values)
            human_state = Variable(human_state)

            self.optimizer.zero_grad()
            mu, sigma = self.source(human_state)
            dist_source = Normal(mu, sigma)
            sample = dist_source.rsample()
            human_next_state = self.transition.select_state(sample, human_state)
            mu_p, sigma_p = self.planning(human_state, human_next_state)
            dist_plan = Normal(mu_p, sigma_p)

            MI = (dist_plan.log_prob(sample) - dist_source.log_prob(sample)).mean()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, values) - MI
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
