import logging
import copy
import torch
import numpy as np
from numpy.linalg import norm
from enum import Enum

from crowd_sim.envs.utils.info import *

class FinalState(Enum):
    ReachGoal = 1
    Collision = 2
    Timeout = 3

class ResultStat:
    def __init__(self, duration, final_state, path_goal=None, cumalative_rewards=None):
        self.duration = duration
        self.final_state = final_state
        self.path_goal = path_goal
        self.cumalative_rewards = cumalative_rewards
        self.human_duration = None
        self.acceleration = None


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
        X = []
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

            if isinstance(info, ReachGoal):
                x = ResultStat(self.env.global_time, FinalState.ReachGoal)
            elif isinstance(info, Collision):
                x = ResultStat(self.env.global_time, FinalState.Collision)
            elif isinstance(info, Timeout):
                x = ResultStat(self.env.time_limit, FinalState.Timeout)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, human_states, rewards, imitation_learning)

            x.speed = [norm(np.array([action.vx, action.vy]), 2) for action in actions]
            x.cumalative_rewards = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)])

            if phase in ['val', 'test']:
                x.human_duration = self.env.global_time + sum([norm(np.array(human.get_position()) - np.array(human.get_goal_position()), 2) / human.v_pref for human in self.env.humans]) / len(self.env.humans)
                x.acceleration = (abs(np.diff(np.asarray(x.speed))) / self.robot.time_step).mean()

            X.append(x)

        success_cases = [1. for x in X if x.final_state == FinalState.ReachGoal]
        collision_cases = [1. for x in X if x.final_state == FinalState.Collision]
        avg_nav_time = sum([x.duration for x in X if x.final_state == FinalState.ReachGoal]) / len(success_cases)
        avg_nav_length = sum([vel * self.robot.time_step for vel in x.speed for x in X if x.final_state == FinalState.ReachGoal]) / len(success_cases)
        avg_cumalative_rewards = average([x.cumalative_rewards for x in X])

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, nav length: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, sum(success_cases) / k, sum(collision_cases) / k, avg_nav_time, avg_nav_length,
                            avg_cumalative_rewards))

        if phase in ['val', 'test']:
            total_time = sum([x.duration for x in X]) * self.robot.time_step
            robot_avg_acc = average([x.acceleration for x in X])
            avg_human_times = average([x.human_duration for x in X])
            logging.info('Frequency of being in danger: {:.2f} and average min separate distance in danger: {:.2f}, '
                         'acceleration: {:.2f}, human nav time: {:.2f}'.format(
                         too_close / total_time, average(min_dist), robot_avg_acc, avg_human_times))

        if print_failure:
            timeout_cases = [1. for x in X if x.final_state == FinalState.Timeout]
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
