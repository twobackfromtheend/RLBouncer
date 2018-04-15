import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bounce

SIDE_WALL_DISTANCE = 4096
BACK_WALL_DISTANCE = 5140
CEILING_DISTANCE = 2044
CORNER_WALL_DISTANCE = 8000
GOAL_X = 892.75
GOAL_Z = 640
x = 590
# CURVE_RADIUS_1, CURVE_RADIUS_2, CURVE_RADIUS_3 = 520, 260, 190  # ramp radii
CURVE_RADIUS_1, CURVE_RADIUS_2, CURVE_RADIUS_3 = x, x / 2, 175  # ramp radii

CURVE_X_1 = SIDE_WALL_DISTANCE - CURVE_RADIUS_1
CURVE_X_2 = SIDE_WALL_DISTANCE - CURVE_RADIUS_2
CURVE_X_3 = SIDE_WALL_DISTANCE - CURVE_RADIUS_3
CURVE_Y_1 = BACK_WALL_DISTANCE - CURVE_RADIUS_1
# CURVE_Y_2 = BACK_WALL_DISTANCE - CURVE_RADIUS_2
CURVE_Y_3 = BACK_WALL_DISTANCE - CURVE_RADIUS_3
CURVE_Z_1 = CEILING_DISTANCE - CURVE_RADIUS_1
CURVE_Z_2 = CURVE_RADIUS_2
CURVE_Z_3 = CURVE_RADIUS_3


class Ball:
    csv_header = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'rotx', 'roty', 'rotz', 'rotvx', 'rotvy', 'rotvz']
    ball_radius = 91.25
    gravity = 650  # uu/s2
    air_resistance = 0.0305  # % loss per second
    ball_max_speed = 6000
    ball_max_rotation_speed = 6

    def __init__(self, file_path, show=True, save=False):
        self.file_path = os.path.join(os.getcwd(), 'data', file_path)
        self.df = pd.read_csv(self.file_path, header=None, names=self.csv_header)
        self.set_simulation_initial_variables()
        self.sim_data = self.predict_ball_positions()
        if show:
            self.plot_sim_data()
        if save:
            self.save_sim_data()

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
        sim_data = self.simulate_time(self.df['t'].min(), self.df['t'].max(), 1 / 120, self.step_dt, starting_x_v)
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
        # ramps
        # bottom y axis
        if x[1] > CURVE_Y_3 and x[2] < CURVE_Z_3 and abs(x[0]) > GOAL_X and \
                (abs(x[1]) - CURVE_Y_3) ** 2 + (x[2] - CURVE_Z_3) ** 2 > (CURVE_RADIUS_3 - self.ball_radius) ** 2:
            surface_vector = np.array([0, CURVE_Y_3 - x[1], CURVE_Z_3 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'y+ bottom')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[1] < -CURVE_Y_3 and x[2] < CURVE_Z_3 and abs(x[0]) > GOAL_X and \
                (abs(x[1]) - CURVE_Y_3) ** 2 + (x[2] - CURVE_Z_3) ** 2 > (CURVE_RADIUS_3 - self.ball_radius) ** 2:
            surface_vector = np.array([0, CURVE_Y_3 - x[1], CURVE_Z_3 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'y- bottom')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        # bottom x axis
        if x[0] > CURVE_X_2 and x[2] < CURVE_Z_2 and \
                (abs(x[0]) - CURVE_X_2) ** 2 + (x[2] - CURVE_Z_2) ** 2 > (CURVE_RADIUS_2 - self.ball_radius) ** 2:
            surface_vector = np.array([CURVE_X_2 - x[0], 0, CURVE_Z_2 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'x+ bottom')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[0] < -CURVE_X_2 and x[2] < CURVE_Z_2 and \
                (abs(x[0]) - CURVE_X_2) ** 2 + (x[2] - CURVE_Z_2) ** 2 > (CURVE_RADIUS_2 - self.ball_radius) ** 2:
            surface_vector = np.array([CURVE_X_2 - x[0], 0, CURVE_Z_2 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'x- bottom')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        # top y axis
        if x[1] > CURVE_Y_1 and x[2] > CURVE_Z_1 and abs(x[0]) > GOAL_X and \
                (abs(x[1]) - CURVE_Y_1) ** 2 + (x[2] - CURVE_Z_1) ** 2 > (CURVE_RADIUS_1 - self.ball_radius) ** 2:
            surface_vector = np.array([0, CURVE_Y_1 - x[1], CURVE_Z_1 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'y+ top')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[1] < -CURVE_Y_1 and x[2] > CURVE_Z_1 and abs(x[0]) > GOAL_X and \
                (abs(x[1]) - CURVE_Y_1) ** 2 + (x[2] - CURVE_Z_1) ** 2 > (CURVE_RADIUS_1 - self.ball_radius) ** 2:
            surface_vector = np.array([0, CURVE_Y_1 - x[1], CURVE_Z_1 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'y- top')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        # top x axis
        if x[0] > CURVE_X_1 and x[2] > CURVE_Z_1 and \
                (abs(x[0]) - CURVE_X_1) ** 2 + (x[2] - CURVE_Z_1) ** 2 > (CURVE_RADIUS_1 - self.ball_radius) ** 2:
            surface_vector = np.array([CURVE_X_1 - x[0], 0, CURVE_Z_1 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'x+ top')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[0] < -CURVE_X_1 and x[2] > CURVE_Z_1 and \
                (abs(x[0]) - CURVE_X_1) ** 2 + (x[2] - CURVE_Z_1) ** 2 > (CURVE_RADIUS_1 - self.ball_radius) ** 2:
            surface_vector = np.array([CURVE_X_2 - x[0], 0, CURVE_Z_1 - x[2]])
            normal_vector = surface_vector / np.sqrt(surface_vector.dot(surface_vector))
            # print(t, x, normal_vector, surface_vector)
            # print(t, 'x- top')
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
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
        if x[1] < -BACK_WALL_DISTANCE + self.ball_radius and \
                self.ball_radius < x[2] < CEILING_DISTANCE - self.ball_radius and \
                (abs(x[0]) > GOAL_X - self.ball_radius or abs(x[2]) > GOAL_Z - self.ball_radius):
            normal_vector = np.array([0, 1, 0])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True
        elif x[1] > BACK_WALL_DISTANCE - self.ball_radius and \
                self.ball_radius < x[2] < CEILING_DISTANCE - self.ball_radius and \
                (abs(x[0]) > GOAL_X - self.ball_radius or abs(x[2]) > GOAL_Z - self.ball_radius):
            normal_vector = np.array([0, -1, 0])
            if self.check_if_ball_leaving(x_v, normal_vector):
                collided = True

        # corner side
        if abs(x[0]) + abs(x[1]) + self.ball_radius > CORNER_WALL_DISTANCE:
            over_rt2 = 1 / np.sqrt(2)
            if x[0] < 0 and x[1] < 0:
                normal_vector = np.array([over_rt2, over_rt2, 0])
                if self.check_if_ball_leaving(x_v, normal_vector):
                    collided = True
            elif x[0] < 0 and x[1] > 0:
                normal_vector = np.array([over_rt2, -over_rt2, 0])
                if self.check_if_ball_leaving(x_v, normal_vector):
                    collided = True
            elif x[0] > 0 and x[1] < 0:
                normal_vector = np.array([-over_rt2, over_rt2, 0])
                if self.check_if_ball_leaving(x_v, normal_vector):
                    collided = True
            elif x[0] > 0 and x[1] > 0:
                normal_vector = np.array([-over_rt2, -over_rt2, 0])
                if self.check_if_ball_leaving(x_v, normal_vector):
                    collided = True


        # # Top Ramp X-axis
        # if abs(x) > wx / 2 - cR and z > cz and (abs(x) - cx) ** 2 + (z - cz) ** 2 > (cR - R) ** 2:
        #     a = math.atan2(z - cz, abs(x) - cx) / pi * 180
        #     return True, [0, (90 + a) * sign(x)]
        #
        # # Top Ramp Y-axis
        # if abs(y) > cy and z > cz and (abs(y) - cy) ** 2 + (z - cz) ** 2 > (cR - R) ** 2:
        #     a = math.atan2(z - cz, abs(y) - cy) / pi * 180
        #     return True, [(90 + a) * sign(y), 0]
        # # Bottom Ramp Y-axis
        # elif abs(y) > cy3 and z < cz3 and abs(x) > gx / 2 - R / 2 and (abs(y) - cy3) ** 2 + (z - cz2) ** 2 > (
        #             cR3 - R) ** 2:
        # a = math.atan2(z - cz2, abs(y) - cy3) / pi * 180
        # return True, [(90 + a) * sign(y), 0]

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

        # if ang_vel > max rotation: normalise to 6
        ang_vel = self.sim_vars['ang_vel']
        if ang_vel.dot(ang_vel) > self.ball_max_rotation_speed ** 2:
            ang_vel = ang_vel / np.sqrt(ang_vel.dot(ang_vel)) * self.ball_max_rotation_speed
            self.sim_vars['ang_vel'] = ang_vel

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
            ax.grid()
            # ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()

    def save_sim_data(self):
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
            ax.grid()
            # ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        save_file_path = os.path.splitext(self.file_path)[0] + '.png'
        plt.savefig(save_file_path, dpi=300)

    def plot_trajectory(self, t, positions):
        plt.plot(t, positions.loc[:, 'z'], '-')
        # plt.plot(t, positions.loc[:, 'y'], '-')
        # plt.show()


def save_all_for_data():
    for file_name in os.listdir(os.path.join(os.getcwd(), 'data')):
        if file_name.endswith('.csv'):
            file_path = os.path.join(os.getcwd(), 'data', file_name)
            print(file_path)
            Ball(file_path, show=False, save=True)

if __name__ == '__main__':
    # file_name = "episode_000008.csv"
    # file_name = "episode_000003.csv"  # y+ bottom ramp
    # file_name = "episode_000012.csv"  # y+ bottom ramp
    # file_name = "episode_000010.csv"  # x- bottom
    # file_name = "episode_000015.csv"  # x- bottom
    # file_name = "episode_000029.csv"  # x- bottom
    # file_name = "episode_000035.csv"  # x- bottom
    # file_path = os.path.join(os.getcwd(), 'data', file_name)
    # Ball(file_path)

    for file_name in os.listdir(os.path.join(os.getcwd(), 'data')):
        if file_name.endswith('.csv'):

            file_path = os.path.join(os.getcwd(), 'data', file_name)
            print(file_path)
            Ball(file_path)
            # x = input('Press enter to continue...')

    # save_all_for_data()