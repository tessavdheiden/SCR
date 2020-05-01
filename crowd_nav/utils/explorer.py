import logging
import copy
import torch
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import *
from crowd_sim.envs.visualization.observer_subscriber import notify
from crowd_sim.envs.visualization.video import Video
from crowd_sim.envs.visualization.plotter import Plotter

def get_speed(actions):
    if isinstance(actions[0], ActionXY):
        return np.asarray([norm(np.array([action.vx, action.vy]), 2) for action in actions])
    else:
        return np.asarray([abs(action.v) for action in actions])

def get_acceleration(speed, delta_t):
    return np.diff(speed) / delta_t

def get_jerk(acceleration, delta_t):
    return np.diff(acceleration) / delta_t

def get_path_length(speed, delta_t):
    return sum(speed) * delta_t


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

    def run_episode(self, phase, video_file=None, plot_file=None):
        if video_file:
            observation_subscribers = []
            video = Video(video_file)
            plotter = Plotter(plot_file)
            observation_subscribers += [video, plotter]

        self.robot.policy.set_phase(phase)

        too_close = 0
        min_dist = []

        ob = self.env.reset(phase)
        done = False
        rewards = []
        actions = []

        while not done:
            action = self.robot.act(ob)
            ob, reward, done, info = self.env.step(action)
            notify(observation_subscribers, self.env.state)

            actions.append(action)
            rewards.append(reward)

            if isinstance(info, Danger):
                too_close += 1
                min_dist.append(info.min_dist)

        if video_file:
            video.make()
        if plot_file:
            plotter.save()
        plt.close()

        avg_rewards = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                           * reward for t, reward in enumerate(rewards)])

        if isinstance(info, ReachGoal):
            human_times = average(self.env.get_human_times())
            extra_info = 'success, avg human time: {:.2f}'.format(human_times)
        elif isinstance(info, Collision):
            extra_info = 'collision'
        elif isinstance(info, Timeout):
            extra_info = 'time out'
        else:
            raise ValueError('Invalid end signal from environment')

        speed = get_speed(actions)
        path_length = get_path_length(speed, self.robot.time_step)
        jerk = sum(get_jerk(get_acceleration(speed, self.robot.time_step), self.robot.time_step)) / len(actions)

        logging.info('{:<5} has {}, nav time: {:.2f}, nav path length: {:.2f}, jerk: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info,  self.env.global_time, path_length, jerk, avg_rewards))

        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                     too_close / self.env.global_time, average(min_dist))


    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            joined_states = []
            actions = []
            rewards = []
            states = []
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)

                joined_states.append(self.robot.policy.last_state)
                states.append(ob + [self.robot.get_observable_state()])
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(joined_states, states, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else f'in episode {episode} '
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, joined_states, states, rewards, imitation_learning=False):
        """
        Updates the memory with experiences.
        :param joined_states: a list of len (# experience in episode, # agents) of rotated states (if not IL)
        :param states: a list of len (# experience in episode, # agents) of observable states
        """
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i in range(len(joined_states)):
            reward = rewards[i]
            joined_state = joined_states[i]
            state = torch.Tensor([(s.px, s.py, s.vx, s.vy, s.radius) for s in states[i]]).to(self.device)

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                joined_state = self.target_policy.transform(joined_state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(joined_states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = joined_states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)
            self.memory.push((joined_state, value, state))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
