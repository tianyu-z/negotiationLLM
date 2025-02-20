# reference: https://github.com/longtermrisk/marltoolbox
import copy
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os

grandparent_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(grandparent_folder)
from utils.io_utils import get_hex_time
from utils.visualize_utils import create_gif


def is_version_below_or_equal_38():
    current_version = sys.version_info
    if current_version.major == 3 and current_version.minor <= 8:
        return True
    elif current_version.major < 3:
        return True
    else:
        return False


# Example usage
if is_version_below_or_equal_38():
    from collections import Iterable
else:
    from collections.abc import Iterable

import gym
import logging
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CoinGame(gym.Env):
    """
    Coin Game environment.
    """

    NAME = "CoinGame"
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = None
    MOVES = [
        np.array([0, 1]),
        np.array([0, -1]),
        np.array([1, 0]),
        np.array([-1, 0]),
    ]

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}

        self._validate_config(config)
        self.save_dir = f"CoinGame_{get_hex_time()}"
        self._load_config(config)
        self.player_red_id, self.player_blue_id = self.players_ids
        self.n_features = self.grid_size**2 * (2 * self.NUM_AGENTS)
        self.OBSERVATION_SPACE = gym.spaces.Box(
            low=0, high=1, shape=(self.grid_size, self.grid_size, 4), dtype="uint8"
        )

        self.step_count_in_current_episode = None
        if self.output_additional_info:
            self._init_info()
        self.seed(seed=config.get("seed", None))

    def _validate_config(self, config):
        if "players_ids" in config:
            assert isinstance(config["players_ids"], Iterable)
            assert len(config["players_ids"]) == self.NUM_AGENTS

    def _load_config(self, config):
        self.players_ids = config.get("players_ids", ["player_red", "player_blue"])
        self.max_steps = config.get("max_steps", 20)
        self.grid_size = config.get("grid_size", 3)
        self.output_additional_info = config.get("output_additional_info", True)
        self.asymmetric = config.get("asymmetric", False)
        self.both_players_can_pick_the_same_coin = config.get(
            "both_players_can_pick_the_same_coin", True
        )

    def seed(self, seed=None):
        """Seed the PRNG of this space."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count_in_current_episode = 0

        if self.output_additional_info:
            self._reset_info()

        self._randomize_color_and_player_positions()
        self._generate_coin()
        obs = self._generate_observation()

        return {self.player_red_id: obs[0], self.player_blue_id: obs[1]}

    def _randomize_color_and_player_positions(self):
        # Reset coin color and the players and coin positions
        self.red_coin = self.np_random.integers(low=0, high=2)
        self.red_pos = self.np_random.integers(low=0, high=self.grid_size, size=(2,))
        self.blue_pos = self.np_random.integers(low=0, high=self.grid_size, size=(2,))
        self.coin_pos = np.zeros(shape=(2,), dtype=np.int8)

        self._players_do_not_overlap_at_start()

    def _players_do_not_overlap_at_start(self):
        while self._same_pos(self.red_pos, self.blue_pos):
            self.blue_pos = self.np_random.integers(self.grid_size, size=2)

    def _generate_coin(self):
        self._switch_between_coin_color_at_each_generation()
        self._coin_position_different_from_players_positions()

    def _switch_between_coin_color_at_each_generation(self):
        self.red_coin = 1 - self.red_coin

    def _coin_position_different_from_players_positions(self):
        success = 0
        while success < self.NUM_AGENTS:
            self.coin_pos = self.np_random.integers(self.grid_size, size=2)
            success = 1 - self._same_pos(self.red_pos, self.coin_pos)
            success += 1 - self._same_pos(self.blue_pos, self.coin_pos)

    def _generate_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 4))
        obs[self.red_pos[0], self.red_pos[1], 0] = 1
        obs[self.blue_pos[0], self.blue_pos[1], 1] = 1
        if self.red_coin:
            obs[self.coin_pos[0], self.coin_pos[1], 2] = 1
        else:
            obs[self.coin_pos[0], self.coin_pos[1], 3] = 1

        obs = self._get_obs_invariant_to_the_player_trained(obs)

        return obs

    def step(self, actions: Dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: observations, rewards, done, info
        """
        actions = self.dictaction2listaction(actions)

        self.step_count_in_current_episode += 1
        self._move_players(actions)
        reward_list, generate_new_coin = self._compute_reward()
        if generate_new_coin:
            self._generate_coin()
        observations = self._generate_observation()
        state, rewards, done, info = self.gather_info(observations, reward_list)
        return state, rewards, done, info

    def _same_pos(self, x, y):
        return (x == y).all()

    def _move_players(self, actions):
        # red player has to be the first one to move
        # blue player has to be the second one to move
        self.red_pos = (self.red_pos + self.MOVES[actions[0]]) % self.grid_size
        self.blue_pos = (self.blue_pos + self.MOVES[actions[1]]) % self.grid_size

    def _compute_reward(self):
        reward_red = 0.0
        reward_blue = 0.0
        generate_new_coin = False
        red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = (
            False,
            False,
            False,
            False,
        )

        red_first_if_both = None
        if not self.both_players_can_pick_the_same_coin:
            if self._same_pos(self.red_pos, self.coin_pos) and self._same_pos(
                self.blue_pos, self.coin_pos
            ):
                red_first_if_both = bool(self.np_random.integers(low=0, high=2))

        if self.red_coin:
            if self._same_pos(self.red_pos, self.coin_pos) and (
                red_first_if_both is None or red_first_if_both
            ):
                generate_new_coin = True
                reward_red += 1
                if self.asymmetric:
                    reward_red += 3
                red_pick_any = True
                red_pick_red = True
            if self._same_pos(self.blue_pos, self.coin_pos) and (
                red_first_if_both is None or not red_first_if_both
            ):
                generate_new_coin = True
                reward_red += -2
                reward_blue += 1
                blue_pick_any = True
        else:
            if self._same_pos(self.red_pos, self.coin_pos) and (
                red_first_if_both is None or red_first_if_both
            ):
                generate_new_coin = True
                reward_red += 1
                reward_blue += -2
                if self.asymmetric:
                    reward_red += 3
                red_pick_any = True
            if self._same_pos(self.blue_pos, self.coin_pos) and (
                red_first_if_both is None or not red_first_if_both
            ):
                generate_new_coin = True
                reward_blue += 1
                blue_pick_blue = True
                blue_pick_any = True

        reward_list = [reward_red, reward_blue]

        if self.output_additional_info:
            self._accumulate_info(
                red_pick_any=red_pick_any,
                red_pick_red=red_pick_red,
                blue_pick_any=blue_pick_any,
                blue_pick_blue=blue_pick_blue,
            )

        return reward_list, generate_new_coin

    def dictaction2listaction(self, actions):
        """
        Format actions from dict of players to list of lists
        """
        actions = [actions[player_id] for player_id in self.players_ids]
        return actions

    def _get_obs_invariant_to_the_player_trained(self, observation):
        """
        We want to be able to use a policy trained as player 1,
        for evaluation as player 2 and vice versa.
        # add the viewpoint from the blue player as the first player
        """

        # player_red_observation contains
        # [Red pos, Blue pos, Red coin pos, Blue coin pos]
        player_red_observation = observation
        # After modification, player_blue_observation will contain
        # [Blue pos, Red pos, Blue coin pos, Red coin pos]
        player_blue_observation = copy.deepcopy(observation)
        player_blue_observation[..., 0] = observation[..., 1]
        player_blue_observation[..., 1] = observation[..., 0]
        player_blue_observation[..., 2] = observation[..., 3]
        player_blue_observation[..., 3] = observation[..., 2]

        return [player_red_observation, player_blue_observation]

    def gather_info(self, observations, rewards):
        state = {
            self.player_red_id: observations[0],
            self.player_blue_id: observations[1],
        }
        rewards = {
            self.player_red_id: rewards[0],
            self.player_blue_id: rewards[1],
        }

        epi_is_done = self.step_count_in_current_episode >= self.max_steps
        if self.step_count_in_current_episode > self.max_steps:
            logger.warning(
                "step_count_in_current_episode > self.max_steps: "
                f"{self.step_count_in_current_episode} > {self.max_steps}"
            )

        done = {
            self.player_red_id: epi_is_done,
            self.player_blue_id: epi_is_done,
            "__all__": epi_is_done,
        }

        if epi_is_done and self.output_additional_info:
            player_red_info, player_blue_info = self._get_episode_info()
            info = {
                self.player_red_id: player_red_info,
                self.player_blue_id: player_blue_info,
            }
        else:
            info = {}

        return state, rewards, done, info

    def _get_episode_info(self):
        """
        Output the following information:
        pick_speed is the fraction of steps during which the player picked a
        coin.
        pick_own_color is the fraction of coins picked by the player which have
        the same color as the player.
        """
        player_red_info, player_blue_info = {}, {}

        if len(self.red_pick) > 0:
            red_pick = sum(self.red_pick)
            player_red_info["pick_speed"] = red_pick / len(self.red_pick)
            if red_pick > 0:
                player_red_info["pick_own_color"] = sum(self.red_pick_own) / red_pick

        if len(self.blue_pick) > 0:
            blue_pick = sum(self.blue_pick)
            player_blue_info["pick_speed"] = blue_pick / len(self.blue_pick)
            if blue_pick > 0:
                player_blue_info["pick_own_color"] = sum(self.blue_pick_own) / blue_pick

        return player_red_info, player_blue_info

    def _reset_info(self):
        self.red_pick.clear()
        self.red_pick_own.clear()
        self.blue_pick.clear()
        self.blue_pick_own.clear()

    def _accumulate_info(
        self, red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue
    ):
        self.red_pick.append(red_pick_any)
        self.red_pick_own.append(red_pick_red)
        self.blue_pick.append(blue_pick_any)
        self.blue_pick_own.append(blue_pick_blue)

    def _init_info(self):
        self.red_pick = []
        self.red_pick_own = []
        self.blue_pick = []
        self.blue_pick_own = []

    def render(self, mode="human", show=False, iter=None):
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(self.grid_size, self.grid_size))
        ax.set_xlim(-1, self.grid_size)
        ax.set_ylim(-1, self.grid_size)

        # Set up grid
        for i in range(self.grid_size + 1):
            ax.axhline(i, lw=2, color="k", zorder=1)
            ax.axvline(i, lw=2, color="k", zorder=1)

        # Set up a colormap for the agents and coins
        # #FF0000 is red, #0000FF is blue, #FFA500 is orange, #00FF00 is green
        cmap = ListedColormap(["#FF0000", "#0000FF"])
        d = 0.25
        # Plot red agent
        red_agent = patches.Rectangle(
            self.red_pos - d / 2, d, d, facecolor=cmap(0), edgecolor="k", zorder=2
        )
        ax.add_patch(red_agent)

        # Plot blue agent
        blue_agent = patches.Rectangle(
            self.blue_pos - d / 2, d, d, facecolor=cmap(1), edgecolor="k", zorder=2
        )
        ax.add_patch(blue_agent)

        # Plot red coin
        if self.red_coin:
            red_coin = patches.Circle(
                self.coin_pos,
                d / 2,
                facecolor=cmap(0),
                edgecolor="k",
                zorder=2,
            )
            ax.add_patch(red_coin)

        # Plot blue coin
        else:
            blue_coin = patches.Circle(
                self.coin_pos,
                d / 2,
                facecolor=cmap(1),
                edgecolor="k",
                zorder=2,
            )
            ax.add_patch(blue_coin)

        # Remove axis labels
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        plt.title(f"Frame_{iter}")
        # rendering
        if show:
            # Show the plot
            plt.show()
        if self.save_dir is not None:
            # Save the plot
            if not os.path.exists(f".log/{self.save_dir}"):
                os.makedirs(f".log/{self.save_dir}")
            if iter is not None:
                fig.savefig(f".log/{self.save_dir}/coin_game_{iter}.png")
            else:
                fig.savefig(".log/{self.save_dir}/coin_game.png")
            return f".log/{self.save_dir}"


class AsymCoinGame(CoinGame):
    NAME = "AsymCoinGame"

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}

        if "asymmetric" in config:
            assert config["asymmetric"]
        else:
            config["asymmetric"] = True
        super().__init__(config)


if __name__ == "__main__":
    # test the environment
    env = CoinGame()
    obs = env.reset()
    config = {
        "players_ids": ["player_red", "player_blue"],
        "max_steps": 20,
        "grid_size": 3,
        "output_additional_info": True,
        "asymmetric": False,
        "both_players_can_pick_the_same_coin": True,
    }
    for i in range(10):
        print("red position", obs["player_red"][:, :, 0])
        print("blue position", obs["player_red"][:, :, 1])
        print("red coin position", obs["player_red"][:, :, 2])
        print("blue coin position", obs["player_red"][:, :, 3])
        output_path = env.render(iter=i, show=False)
        action1, action2 = env.ACTION_SPACE.sample(), env.ACTION_SPACE.sample()
        obs, reward, done, info = env.step(
            {"player_red": action1, "player_blue": action2}
        )
    create_gif(output_path, os.path.join(output_path, "coin_game.gif"))
    env.close()
