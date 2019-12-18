import logging
import copy
import torch
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.info import *


class ResultStat(object):
    def __init__(self, duration: float, cumulative_rewards: float, epoch: int):
        self.duration = duration
        self.cumulative_rewards = cumulative_rewards
        self.epoch = epoch


class RSSuccess(ResultStat):
    def __init__(self, duration: float, cumulative_rewards: float, epoch: int, speed, delta_t: float):
        super().__init__(duration, cumulative_rewards, epoch)
        self.duration = duration
        self.cumulative_rewards = cumulative_rewards
        self.epoch = epoch
        self.speed = speed
        self.delta_t = delta_t
        self.avg_human_time = None

    @property
    def acceleration(self):
        return np.diff(self.speed) / self.delta_t

    @property
    def jerk(self):
        return np.diff(self.acceleration) / self.delta_t

    def human_times(self, humans, global_time):
        self.avg_human_time = global_time + sum([norm(np.array(human.get_position()) - np.array(human.get_goal_position()), 2) / human.v_pref for human in humans]) / len(humans)


class RSCollision(ResultStat):
    def __init__(self, duration: float, cumulative_rewards: float, epoch: int):
        super().__init__(duration, cumulative_rewards, epoch)
        self.duration = duration
        self.cumulative_rewards = cumulative_rewards
        self.epoch = epoch


class RSTimeOut(ResultStat):
    def __init__(self, duration: float, cumulative_rewards: float, epoch: int):
        super().__init__(duration, cumulative_rewards, epoch)
        self.duration = duration
        self.cumulative_rewards = cumulative_rewards
        self.epoch = epoch


class TestData(object):
    def __init__(self, phase):
        self.phase = phase
        self.results = []

    def push(self, result: ResultStat, humans: list, global_time: float):
        if isinstance(result, RSSuccess):
            if self.phase in ['val', 'test']:
                result.human_times(global_time=global_time, humans=humans)
                self.results.append(result)
            else:
                self.results.append(result)

    @property
    def size(self):
        return len(self.results)


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)

        X = TestData(phase)
        too_close = 0
        min_dist = []
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            human_states = []
            rewards = []
            while not done:
                action = self.robot.act(ob)
                human_state, reward, done, info = self.env.step(action)
                actions.append(action)
                states.append(self.robot.policy.last_state)
                human_states.append(human_state)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            cumalative_rewards = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                            * reward for t, reward in enumerate(rewards)])

            if isinstance(info, ReachGoal):
                speed = np.asarray([norm(np.array([action.vx, action.vy]), 2) for action in actions])
                x = RSSuccess(self.env.global_time, cumalative_rewards, i, speed, self.robot.time_step)
            elif isinstance(info, Collision):
                x = RSCollision(self.env.global_time, cumalative_rewards, i)
            elif isinstance(info, Timeout):
                x = RSTimeOut(self.env.time_limit, cumalative_rewards, i)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, human_states, rewards, imitation_learning)

            X.push(x, global_time=self.env.global_time, humans=self.env.humans)

        success_cases = [x.epoch for x in X.results if isinstance(x, RSSuccess)] #reduce(lambda a, _: a + 1, (x for x in test_results if isinstance(x, RSSuccess)), 0)
        collision_cases = [x.epoch for x in X.results if isinstance(x, RSCollision)]
        avg_nav_time = sum([x.duration for x in X.results if isinstance(x, RSSuccess)]) / len(success_cases) if len(success_cases) else 0

        avg_nav_length = sum([vel * self.robot.time_step for vel in x.speed for x in X.results if isinstance(x, RSSuccess)]) / len(success_cases) if len(success_cases) else 0
        avg_cumulative_rewards = average([x.cumulative_rewards for x in X.results])

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, nav length: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, len(success_cases) / k, len(collision_cases) / k, avg_nav_time, avg_nav_length,
                            avg_cumulative_rewards))

        if phase in ['val', 'test']:
            total_time = sum([x.duration for x in X.results]) * self.robot.time_step
            avg_jerk = average([x.jerk.mean() for x in X.results if isinstance(x, RSSuccess)])
            avg_human_times = average([x.avg_human_time for x in X.results if isinstance(x, RSSuccess)])
            logging.info('Frequency of being in danger: {:.2f} and average min separate distance in danger: {:.2f}, '
                         'jerk: {:.4f}, human nav time: {:.2f}'.format(
                         too_close / total_time, average(min_dist), avg_jerk, avg_human_times))

        if print_failure:
            timeout_cases = [x.epoch for x in X.results if isinstance(x.info, Timeout)]
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, human_states, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                h_s = torch.Tensor([(s.px, s.py, s.vx, s.vy, s.radius) for s in state.human_states]).to(
                    self.device)

                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                h_s = torch.Tensor([(s.px, s.py, s.vx, s.vy, s.radius) for s in human_states[i]]).to(
                    self.device)

                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])

            self.memory.push((state, value, h_s))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
