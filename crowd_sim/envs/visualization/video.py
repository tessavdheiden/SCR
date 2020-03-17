import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.lines as mlines
from matplotlib import patches
from matplotlib import animation

from crowd_sim.envs.visualization.observer_subscriber import ObservationSubscriber


class Video(ObservationSubscriber):
    def __init__(self, file):
        self.file = file
        self.frames = []
        self.attention_weights = None
        self.human_num = None
        self.artists = list()

    def on_observation(self, observation):
        frame = self.extract_frame(observation)
        self.frames.append(frame)

    def extract_frame(self, observation):
        return observation

    def make_ax(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        self.ax = ax
        self.fig = fig

    def add_robot_circle(self):
        position0 = self.frames[0][0].position
        radius = self.frames[0][0].radius
        self.robot = plt.Circle(position0, radius, fill=True, fc='orange', color='k')
        self.ax.add_artist(self.robot)

    def add_human_circles(self):
        x_offset = 0.11
        y_offset = 0.11
        self.human_num = len(self.frames[0][1])
        positions0 = [self.frames[0][1][i].position for i in range(self.human_num)]
        self.radii = [self.frames[0][1][i].radius for i in range(self.human_num)]
        self.humans = [plt.Circle(position, self.radii [i], fill=False) for i, position in enumerate(positions0)]
        self.human_numbers = [plt.text(self.humans[i].center[0] - x_offset, self.humans[i].center[1] - y_offset, str(i),
                                  color='black', fontsize=12) for i in range(self.human_num)]

        for i, human in enumerate(self.humans):
           self.ax.add_artist(human)
           self.ax.add_artist(self.human_numbers[i])

    def add_goal(self):
        goal = mlines.Line2D([0], [4], color='red', marker='*', linestyle='None', markersize=15, label='Goal')
        self.ax.add_artist(goal)

    def add_robot_orientation(self):
        theta = np.arctan2(self.frames[0][0].vy, self.frames[0][0].vx)
        orientation = ((0, 0), (self.frames[0][0].radius * np.cos(theta), self.frames[0][0].radius * np.sin(theta)))
        self.robot_ori = patches.FancyArrowPatch(*orientation, color='k', arrowstyle=patches.ArrowStyle("->", head_length=4, head_width=2))
        self.ax.add_artist(self.robot_ori)

    def add_human_orientations(self):
        self.human_ori = [None] * self.human_num
        for i, human in enumerate(self.humans):
            theta = np.arctan2(self.frames[0][1][i].vy, self.frames[0][1][i].vx)
            orientation = ((0, 0), (self.frames[0][1][i].radius * np.cos(theta), self.frames[0][1][i].radius * np.sin(theta)))
            self.human_ori[i] = patches.FancyArrowPatch(*orientation, color='k',
                                                     arrowstyle=patches.ArrowStyle("->", head_length=4, head_width=2))
            self.ax.add_artist(self.human_ori[i])

    def add_text(self):
        self.text = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
        self.ax.add_artist(self.text)

    def make(self, draw_func=None):
        self.make_ax()
        self.add_robot_circle()
        self.add_goal()
        self.add_human_circles()
        self.add_robot_orientation()
        self.add_human_orientations()
        self.add_text()
        self.artists.append(draw_func(self.ax, self.frames[0], True))

        time_step = .25

        def update(frame_num):
            position = self.frames[frame_num][0].position
            theta = np.arctan2(self.frames[frame_num][0].vy, self.frames[frame_num][0].vx)
            self.robot.center = position
            r = mpl.transforms.Affine2D().rotate(theta)
            t = mpl.transforms.Affine2D().translate(position[0], position[1])
            tra = r + t + self.ax.transData
            self.robot_ori.set_transform(tra)

            for i, human in enumerate(self.humans):
                human.center = self.frames[frame_num][1][i].position
                self.human_numbers[i].set_position((human.center[0] - .11, human.center[1] - .11))
                theta = np.arctan2(self.frames[frame_num][1][i].vy, self.frames[frame_num][1][i].vx)
                r = mpl.transforms.Affine2D().rotate(theta)
                t = mpl.transforms.Affine2D().translate(self.frames[frame_num][1][i].px, self.frames[frame_num][1][i].py)
                tra = r + t + self.ax.transData
                self.human_ori[i].set_transform(tra)

            self.text.set_text('Time: {:.2f}'.format(frame_num * time_step))

            if draw_func:
                draw_func(self.ax, self.frames[frame_num])

        def plot_value_heatmap():
            assert self.robot.kinematics == 'holonomic'
            for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                         agent.vx, agent.vy, agent.theta))
            # when any key is pressed draw the action value plot
            fig, axis = plt.subplots()
            speeds = [0] + self.robot.policy.speeds
            rotations = self.robot.policy.rotations + [np.pi * 2]
            r, th = np.meshgrid(speeds, rotations)
            z = np.array(self.action_values[global_step % len(self.states)][1:])
            z = (z - np.min(z)) / (np.max(z) - np.min(z))
            z = np.reshape(z, (16, 5))
            polar = plt.subplot(projection="polar")
            polar.tick_params(labelsize=16)
            mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
            plt.plot(rotations, r, color='k', ls='none')
            plt.grid()
            cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
            cbar = plt.colorbar(mesh, cax=cbaxes)
            cbar.ax.tick_params(labelsize=16)
            plt.show()

        def on_click(event):
            anim.running ^= True
            if anim.running:
                anim.event_source.stop()
                if hasattr(self.robot.policy, 'action_values'):
                    plot_value_heatmap()
            else:
                anim.event_source.start()

        episode_length = len(self.frames)
        self.fig.canvas.mpl_connect('key_press_event', on_click)
        anim = animation.FuncAnimation(self.fig, update, frames=episode_length, interval=time_step * 1000)
        anim.running = True

        if self.file is not None:
            ffmpeg_writer = animation.writers['ffmpeg']
            writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(self.file, writer=writer)

