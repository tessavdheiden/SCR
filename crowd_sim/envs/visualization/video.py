import numpy as np
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

    def on_observation(self, observation):
        frame = self.extract_frame(observation)
        self.frames.append(frame)

    def extract_frame(self, observation):
        return observation

    def make(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        x_offset = 0.11
        y_offset = 0.11
        goal_color = 'red'
        robot_color = 'yellow'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        arrow_color = 'red'

        # add robot and its goal
        robot_positions = [states[0].position + (states[0].theta, states[0].vx, states[0].vy) for states in self.frames]
        robot_radius = self.frames[0][0].radius
        goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        robot = plt.Circle(robot_positions[0], robot_radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        ax.add_artist(goal)
        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

        # add humans and their numbers
        human_positions = [[state.position + (state.theta, state.vx, state.vy) for state in states[1]] for states in self.frames]
        human_num = len(human_positions[0])
        human_radii = [state.radius for state in self.frames[0][1]]
        humans = [plt.Circle(human_positions[0][i], human_radii[i], fill=False)
                  for i in range(human_num)]
        human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                  color='black', fontsize=12) for i in range(human_num)]
        for i, human in enumerate(humans):
            ax.add_artist(human)
            ax.add_artist(human_numbers[i])

        # add time annotation
        time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
        ax.add_artist(time)

        # compute attention scores
        if self.attention_weights is not None:
            attention_scores = [
                plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                         fontsize=16) for i in range(human_num)]

        # compute orientation in each step and use arrow to show the direction
        orientations = []
        for i in range(human_num + 1):
            orientation = []
            for state in self.frames:
                agent_state = state[0] if i == 0 else state[1][i - 1]
                radius = robot_radius if i == 0 else human_radii[i - 1]

                theta = np.arctan2(agent_state.vy, agent_state.vx)
                orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                       agent_state.py + radius * np.sin(theta))))
            orientations.append(orientation)
        arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                  for orientation in orientations]
        for arrow in arrows:
            ax.add_artist(arrow)
        global_step = 0
        time_step = .25

        def update(frame_num):
            nonlocal global_step
            nonlocal arrows
            global_step = frame_num
            robot.center = robot_positions[frame_num]
            for i, human in enumerate(humans):
                human.center = human_positions[frame_num][i]
                human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                for arrow in arrows:
                    arrow.remove()
                arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                  arrowstyle=arrow_style) for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)
                if self.attention_weights is not None:
                    human.set_color(str(self.attention_weights[frame_num][i]))
                    attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

            time.set_text('Time: {:.2f}'.format(frame_num * time_step))

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
        fig.canvas.mpl_connect('key_press_event', on_click)
        anim = animation.FuncAnimation(fig, update, frames=episode_length, interval=time_step * 1000)
        anim.running = True

        if self.file is not None:
            ffmpeg_writer = animation.writers['ffmpeg']
            writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(self.file, writer=writer)

