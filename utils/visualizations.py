import sys
import numpy as np
import torch as th
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def view_train(
        observations, targets, predictions,
        width=4, height=1.3, ego_color="red", obs_color="blue", save=True, save_path="./"
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(visible=True, linewidth=0.2)

    ego = plt.Rectangle(
        xy=[-width/2, -height/2],
        width=width, height=height,
        angle=np.rad2deg(observations[4]), rotation_point="center", facecolor=ego_color
    )
    ax.add_artist(ego)

    obstacle_pos = observations[5:]
    obstacle_pos = th.stack([obstacle_pos[0::5], obstacle_pos[1::5], obstacle_pos[4::5]], dim=-1)
    obstacles = [
        plt.Rectangle(
            xy=[center[0] - width/2, center[1] - height/2],
            width=width, height=height,
            angle=np.rad2deg(center[2]), rotation_point="center", facecolor=obs_color
        )
        for center in obstacle_pos
    ]
    for ob in obstacles:
        ax.add_artist(ob)

    ax.plot(targets[:, 0], targets[:, 1], lw=0.7, color="green", label="true")
    ax.plot(predictions[:, 0], predictions[:, 1], lw=0.7, color="red", label="pred")
    
    ax.legend()
    ax.set_ylim(ax.get_ylim()[::-1])

    plt.axis([-31, 80, -18, 18])
    if save:
        plt.savefig(save_path)
    plt.close()
    

class PlotNeural:
    def __init__(self, figsize=(6, 6)):
        self.world_min_x, self.world_max_x = -5, 50

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def step(self, neural_output):
        vb, yb = neural_output[..., :4], neural_output[..., 4:8]
        self.ax.clear()

        dfs = []
        x_name = r'$v_{d}$  [m/s]'
        y_name = r'$y_{d}$ [m]'
        for i in range(4):
            df = pd.DataFrame({x_name: vb[:, i], y_name: yb[:, i]})
            df['name'] = f"P_{i}"
            dfs.append(df)

        data = pd.concat(dfs)
        sns.kdeplot(data, x=x_name, y=y_name, hue='name', fill=True, ax=self.ax)

        self.ax.set_xlim([0, 30])
        self.ax.set_ylim([-16, 16])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()   

        
class PlotEnv:
    WIDTH, HEIGHT = 4, 1.3
    LANE_WIDTH = 4

    def __init__(self, obstacles, batch_size=10, figsize=(10, 4), ego_color="red", obs_color="blue"):
        self.world_min_x, self.world_max_x = -5, 50

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.grid(visible=True, linewidth=0.2)

        self.ego = plt.Rectangle(
            [-self.WIDTH/2, -self.HEIGHT/2],
            width=self.WIDTH, height=self.HEIGHT,
            angle=0, rotation_point="center", facecolor=ego_color
        )
        self.ax.add_artist(self.ego)

        self.obstacles = [
            plt.Rectangle(
                [center[0] - self.WIDTH/2, center[1] - self.HEIGHT/2],
                width=self.WIDTH, height=self.HEIGHT,
                angle=np.rad2deg(center[2]), rotation_point="center", facecolor=obs_color
            )
            for center in obstacles
        ]
        for ob in self.obstacles:
            self.ax.add_artist(ob)

        self.lines = self.ax.plot(*[np.empty((0, 1)) for _ in range(2 * batch_size)], lw=0.7)
        self.lanes = self.ax.plot(*[np.empty((0, 1)) for _ in range(2 * 5)], lw=0.7)

        plt.axis([-21, 90, -18, 18])
        self.ax.set_ylim(self.ax.get_ylim()[::-1])

    def step(self, ego_obs, obs_obs, lanes, trajs, trajs_weight=None, opt_traj=None):
        self.ego.set(angle=np.rad2deg(ego_obs[2]))
        for (rec, obs) in zip(self.obstacles, obs_obs):
            rec.set(
                xy=[obs[0] - self.WIDTH/2, obs[1] - self.HEIGHT/2],
                angle=np.rad2deg(obs[2])
            )

        for i in range(trajs.shape[0]):
            alpha, lw, col = 0.5, 0.3, "black"
            if trajs_weight is None:
                self.lines[i].set_data(trajs[i, :, 0], trajs[i, :, 1])
                self.lines[i].set(alpha=alpha, linewidth=lw, zorder=0)

                if opt_traj is not None:
                    self.lines[0].set_data(opt_traj[0, :], opt_traj[1, :])
                    self.lines[0].set(color=col, alpha=1, linewidth=2.0, zorder=1)

            else:
                k = trajs_weight[i]
                n = len(self.lines)
                self.lines[k].set_data(trajs[k, :, 0], trajs[k, :, 1])

                if i == 0:
                    self.lines[k].set(color=col, alpha=1, linewidth=2.0, zorder=1)
                else:
                    self.lines[k].set(color=np.random.random(3), alpha=alpha, linewidth=lw, zorder=0)

        x_lane = np.arange(-100, 100)
        for i in range(5):
            y_lane = (lanes[1] + 3.7*i) * np.ones_like(x_lane)
            self.lanes[i].set_data(x_lane, y_lane)
            self.lanes[i].set(zorder=-1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()
        