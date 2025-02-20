import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

grandparent_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(grandparent_folder)
from utils.io_utils import get_hex_time


class carbon(gym.Env):
    def __init__(self, num_iterations, random_rest=False, a=1, b=-1, c=2, d=0):
        super(carbon, self).__init__()
        self.save_dir = f"carbon{get_hex_time()}"
        self.num_iterations = num_iterations
        self.current_iteration = 0

        self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))
        self.observation_space = spaces.Tuple((spaces.Box(0, 1), spaces.Box(0, 1)))
        self.history = []

    def reset(self):
        self.current_iteration = 0
        self.history = []

    def step(self, actions):
        pass
        return actions, payoffs, done, False, info

    def render(self, mode="human", iter=None, show=False):
        pass

    def plot_rewards(self, iter=None, show=False, save_dir=None):
        fig, ax = plt.subplots()
        ax.plot(self.cumulative_rewards[0], label="Agent 1")
        ax.plot(self.cumulative_rewards[1], label="Agent 2")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()
        ax.set_title(f"Cumulative Reward Curve:{iter}")
        if show:
            # Show the plot
            plt.show()
        if save_dir is not None:
            # Save the plot
            if not os.path.exists(f".log/{save_dir}"):
                os.makedirs(f".log/{save_dir}")
            if iter is not None:
                fig.savefig(f".log/{save_dir}/coin_game_{iter}.png")
            else:
                fig.savefig(".log/{save_dir}/coin_game.png")
            return f".log/{save_dir}"
