import matplotlib.pylab as plt

from crowd_sim.envs.visualization.observer_subscriber import ObservationSubscriber


class Plotter(ObservationSubscriber):
    def __init__(self, file):
        self.file = file
        self.point_list = []

    def on_observation(self, observation):
        point = self.get_point(observation)
        self.point_list.append(point)

    def get_point(self, observation):
        return observation

    def save(self):
        cmap = plt.cm.get_cmap('hsv', 10)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        x_offset = 0.11
        y_offset = 0.11

        robot_color = 'yellow'

        human_positions = [[state.position for state in states[1]] for states in self.point_list]
        human_num = len(human_positions[0])
        human_radii = [state.radius for state in self.point_list[0][1]]

        robot_positions = [states[0].position for states in self.point_list]
        robot_radius = self.point_list[0][0].radius

        episode_length = len(self.point_list)
        time_step = .25

        for k in range(episode_length):
            if k % 4 == 0 or k == episode_length - 1:
                robot = plt.Circle(robot_positions[k], robot_radius, fill=True, color=robot_color)
                humans = [plt.Circle(human_positions[k][i], human_radii[i], fill=False, color=cmap(i))
                          for i in range(len(human_positions[0]))]
                ax.add_artist(robot)
                for human in humans:
                    ax.add_artist(human)
            # add time annotation
            global_time = k * time_step
            if global_time % 4 == 0 or k == episode_length - 1:
                agents = humans + [robot]
                times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                  '{:.1f}'.format(global_time),
                                  color='black', fontsize=14) for i in range(human_num + 1)]
                for time in times:
                    ax.add_artist(time)
            if k != 0:
                nav_direction = plt.Line2D((robot_positions[k - 1][0], robot_positions[k][0]),
                                           (robot_positions[k - 1][0], robot_positions[k][0]),
                                           color=robot_color, ls='solid')
                human_directions = [plt.Line2D((human_positions[k - 1][i][1], human_positions[k][i][1]),
                                               (human_positions[k - 1][i][1], human_positions[k][i][1]),
                                               color=cmap(i), ls='solid')
                                    for i in range(human_num)]
                ax.add_artist(nav_direction)
                for human_direction in human_directions:
                    ax.add_artist(human_direction)
        plt.legend([robot], ['Robot'], fontsize=16)
        plt.show(block=False)
        plt.savefig(self.file)


