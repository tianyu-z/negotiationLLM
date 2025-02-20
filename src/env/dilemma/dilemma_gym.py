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


class IteratedPrisonersDilemmaEnv(gym.Env):
    def __init__(self, num_iterations, random_rest=False, a=1, b=-1, c=2, d=0):
        super(IteratedPrisonersDilemmaEnv, self).__init__()
        self.save_dir = f"IPD_{get_hex_time()}"
        self.num_iterations = num_iterations
        self.current_iteration = 0

        self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2)))
        self.observation_space = spaces.Tuple((spaces.Box(0, 1), spaces.Box(0, 1)))

        self.payoff_matrix = {
            (0, 0): (a, a),  # (1, 1)
            (0, 1): (b, c),  # (-1, 2)
            (1, 0): (c, b),  # (2, -1)
            (1, 1): (d, d),  # (0, 0)
        }
        self.a, self.b, self.c, self.d = a, b, c, d
        self.action_count = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0,
        }
        self.cumulative_rewards = [[0], [0]]
        self.random_rest = random_rest
        self.history = []

    def analyticcontinuation(self, p0, p1, id):
        return (
            (self.d - self.c - self.b + self.a) * p0 * p1
            + (self.c - self.a) * p0 ** (1 - id) * p1**id
            + (self.b - self.a) * p1 ** (1 - id) * p0**id
            + self.a
        )

    def reset(self):
        self.current_iteration = 0
        self.history = []
        self.cumulative_rewards = [[0], [0]]
        self.action_count = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0,
        }
        if self.random_rest:
            return {
                "state": [self.action_space.sample()] * 2,
                "obs": np.array(self.observation_space.sample()).reshape(-1),
            }
        else:
            return {"state": [0, 0], "obs": np.array([0, 0])}

    def step(self, actions):
        self.current_iteration += 1
        payoffs = self.payoff_matrix[tuple(actions)]

        done = self.current_iteration >= self.num_iterations

        info = {"payoffs": payoffs}

        self.action_count[tuple(actions)] += 1
        self.cumulative_rewards[0].append(self.cumulative_rewards[0][-1] + payoffs[0])
        self.cumulative_rewards[1].append(self.cumulative_rewards[1][-1] + payoffs[1])

        self.history.append(list(actions))
        info["history"] = self.history
        return actions, payoffs, done, info

    def render(self, mode="human", iter=None, show=False):
        if mode == "human":
            total_actions = sum(self.action_count.values())
            print(f"Iteration: {self.current_iteration}")
            print("Action ratios:")
            print(f"  CC: {self.action_count[(0, 0)] / total_actions:.2%}")
            print(f"  CD: {self.action_count[(0, 1)] / total_actions:.2%}")
            print(f"  DC: {self.action_count[(1, 0)] / total_actions:.2%}")
            print(f"  DD: {self.action_count[(1, 1)] / total_actions:.2%}")
            self.plot_rewards(iter=iter, show=show, save_dir=self.save_dir)
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported.")

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


class messagedIteratedPrisonersDilemmaEnv(IteratedPrisonersDilemmaEnv):
    def __init__(self, num_iterations, random_rest=False, len_message=10):
        super(messagedIteratedPrisonersDilemmaEnv, self).__init__(
            num_iterations, random_rest
        )
        self.action_space = spaces.Tuple(
            (spaces.Discrete(2), spaces.Discrete(len_message))
        )  # Each action now consists of a tuple


if __name__ == "__main__":
    env = IteratedPrisonersDilemmaEnv(10)
    env.reset()

    for _ in range(10):
        action1, action2 = env.action_space.sample(), env.action_space.sample()
        print("action1", action1)
        print("action2", action2)
        obs, reward, done, info = env.step([action1, action2])
        print("obs", obs)
        print("reward", reward)
        print("done", done)
        print("info", info)
        if done:
            # env.render()
            break

    env.close()
    # check analytic continuation
    bools = []
    for p0 in [0, 1]:
        for p1 in [0, 1]:
            for id in [0, 1]:
                assert (
                    env.analyticcontinuation(p0, p1, id)
                    == env.payoff_matrix[(p0, p1)][id]
                ), f"analytic continuation is not passed at {p0, p1, id}"
                bools.append(
                    env.analyticcontinuation(p0, p1, id)
                    == env.payoff_matrix[(p0, p1)][id]
                )
    print("analytic continuation is passed:", np.array(bools).all())
