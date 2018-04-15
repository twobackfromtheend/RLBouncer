import os
import numpy as np
import pandas as pd
import scipy.integrate as spi
import matplotlib.pyplot as plt

import bounce

SIDE_WALL_DISTANCE = 4096
BACK_WALL_DISTANCE = 5120
CEILING_DISTANCE = 2044

class Ball:
    csv_header = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'rotx', 'roty', 'rotz', 'rotvx', 'rotvy', 'rotvz']
    ball_radius = 91.25
    gravity = 650  # uu/s2
    air_resistance = 0.0305  # % loss per second
    ball_max_speed = 6000

    def __init__(self, file_path):
        self.file_path = os.path.join(os.getcwd(), 'data', file_name)
        self.df = pd.read_csv(self.file_path, header=None, names=self.csv_header)
        self.set_simulation_initial_variables()
        self.sim_data = self.predict_ball_positions()
        self.plot_sim_data()

    def set_simulation_initial_variables(self):
        self.sim_vars = {}
        self.sim_vars['position'] = self.df.loc[0, ['x', 'y', 'z']].values
        self.sim_vars['velocity'] = self.df.loc[0, ['vx', 'vy', 'vz']].values
        self.sim_vars['rotation'] = self.df.loc[0, ['rotx', 'roty', 'rotz']].values
        self.sim_vars['ang_vel'] = self.df.loc[0, ['rotvx', 'rotvy', 'rotvz']].values

    def predict_ball_positions(self):
        """
        Returns an array of time and position and velocity up to time=t.
        :param t: Number of seconds to simulate
        :return:
        """
        starting_x_v = np.concatenate((self.sim_vars['position'], self.sim_vars['velocity']))
        sim_data = self.simulate_time(self.df['t'].min(), self.df['t'].max(), 1 / 61.133, self.step_dt, starting_x_v)
        return sim_data

    def simulate_time(self, start_time, end_time, time_step, step_func, starting_values):
        t_s = []
        x_vs = []
        av_s = []

        simulated_time = start_time
        latest_x_v = starting_values
        while simulated_time < end_time:
            # record values at current time
            t_s.append(simulated_time)
            x_vs.append(latest_x_v)
            av_s.append(self.sim_vars['ang_vel'])

            # move by dt
            derivatives = step_func(latest_x_v, simulated_time)
            latest_x_v = latest_x_v + derivatives * time_step
            simulated_time += time_step

        t_s = np.array(t_s)
        x_vs = np.array(x_vs)
        av_s = np.array(av_s)
        sim_data = pd.DataFrame(
            data=np.column_stack((t_s, x_vs, av_s)),
            columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'rotvx', 'rotvy', 'rotvz']
        )
        return sim_data

    def step_dt(self, x_v, t):
        x = x_v[:3]
        v = x_v[3:]
        # calculate collisions
        collided = False
        if x[2] < self.ball_radius:
            # floor
            normal_vector = np.array([0, 0, 1])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[2] > CEILING_DISTANCE - self.ball_radius:
            # ceiling
            normal_vector = np.array([0, 0, -1])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        # sides
        if x[0] < -SIDE_WALL_DISTANCE + self.ball_radius:
            normal_vector = np.array([1, 0, 0])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[0] > SIDE_WALL_DISTANCE - self.ball_radius:
            normal_vector = np.array([-1, 0, 0])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        # back
        if x[1] < -BACK_WALL_DISTANCE + self.ball_radius:
            normal_vector = np.array([0, 1, 0])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[1] > BACK_WALL_DISTANCE - self.ball_radius:
            normal_vector = np.array([0, -1, 0])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True

        if collided:
            state = (v, self.sim_vars['ang_vel'])
            new_state = bounce.bounce(state, normal_vector)
            v, self.sim_vars['ang_vel'] = new_state
            x_v[3:] = v

        # calculate a
        a = np.array([0, 0, -self.gravity]) - self.air_resistance * v

        # if v > max speed: v = v
        if v.dot(v) > self.ball_max_speed ** 2:
            v = v / np.sqrt(v.dot(v)) * self.ball_max_speed
        return np.concatenate((v, a))

    def check_if_ball_leaving(self, x_v, normal_vector):
        if normal_vector.dot(x_v[3:6]) < 0:
            return True
        else:
            return False

    def plot_sim_data(self):
        fig, axes = plt.subplots(3, 3)
        fig.set_size_inches(10, 6)
        axes = axes.flatten()
        axis_plots = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'rotvx', 'rotvy', 'rotvz']
        for i in range(len(axes)):
            ax = axes[i]
            ax.set_title(axis_plots[i])

            ax.plot(self.df.loc[:, 't'], self.df.loc[:, axis_plots[i]], 'k.', ms=1)
            ax.plot(self.sim_data.loc[:, 't'], self.sim_data.loc[:, axis_plots[i]], 'r.', ms=1, alpha=0.7)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_trajectory(self, t, positions):
        plt.plot(t, positions.loc[:, 'z'], '-')
        # plt.plot(t, positions.loc[:, 'y'], '-')
        # plt.show()


if __name__ == '__main__':
    for file_name in os.listdir(os.path.join(os.getcwd(), 'data')):
        file_path = os.path.join(os.getcwd(), 'data', file_name)
        Ball(file_path)
        x = input('Press enter to continue...')