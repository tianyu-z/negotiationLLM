import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
import random


class SimultaneousExchangeEnv(gym.Env):
    def __init__(
        self,
        nb_agents=2,
        max_turns={"lower": 4, "mean": 7, "upper": 10},
        nb_items=3,
        item_max_quantity=5,
        utility_max=10,
        device=None,
    ):
        super(SimultaneousExchangeEnv, self).__init__()

        self.nb_items = nb_items
        self.item_max_quantity = item_max_quantity
        self.utility_max = utility_max
        self._max_turns = max_turns
        self.nb_agents = nb_agents

        # Define action spaces for both cheap talk and actual proposals
        self.action_space = spaces.Dict(
            {
                "cheap_talk": spaces.MultiDiscrete(
                    [self.item_max_quantity + 1] * self.nb_items, dtype=np.int64
                ),
                "proposal": spaces.MultiDiscrete(
                    [self.item_max_quantity + 1] * self.nb_items, dtype=np.int64
                ),
            }
        )

        # Observation space includes:
        # - Item pool quantities
        # - Agent's own utilities
        # - Other agent's cheap talk from previous round
        obs_dict = {
            "item_and_utility": spaces.MultiDiscrete(
                [self.item_max_quantity + 1] * self.nb_items
                + [self.utility_max + 1] * self.nb_items,
                dtype=np.int64,
            ),
            "other_cheap_talk": spaces.MultiDiscrete(
                [self.item_max_quantity + 1] * self.nb_items, dtype=np.int64
            ),
        }
        for i in range(self.nb_agents):
            obs_dict[f"historical_action_agent_{i}"] = spaces.Sequence(
                spaces.MultiDiscrete(
                    [self.item_max_quantity + 1] * self.nb_items, dtype=np.int64
                )
            )
        self.observation_space = spaces.Dict(obs_dict)

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def reset(self):
        """Reset the environment for vectorized operation"""
        # Initialize item pool
        self.item_pool = np.random.randint(
            0, self.item_max_quantity + 1, size=self.nb_items, dtype=np.int64
        )

        # Initialize utilities for both agents
        self.utilities = np.random.randint(
            0,
            self.utility_max + 1,
            size=(self.nb_agents, self.nb_items),
            dtype=np.int64,
        )

        # Initialize game state
        self.turn = 0
        self.next_phase = "None"
        self.is_turn_plus_one = False
        self.phase = "cheap_talk"
        self.rewards = np.zeros(self.nb_agents, dtype=np.int64)
        self.done = False

        # Initialize cheap talk storage with proper shapes
        self.cheap_talk_proposals = [None] * self.nb_agents
        self.actual_proposals = [None] * self.nb_agents
        self.waiting_for_agent = 0

        # Set max turns based on configuration
        if isinstance(self._max_turns, int):
            self.max_turns = self._max_turns
        elif isinstance(self._max_turns, dict):
            lower, mean, upper = (
                self._max_turns["lower"],
                self._max_turns["mean"],
                self._max_turns["upper"],
            )
            self.max_turns = np.random.poisson(mean, size=1)[0]
            self.max_turns = np.clip(self.max_turns, lower, upper)

        # Initialize info dictionary with proper array shapes
        self.info = {
            "turn": self.turn,
            "phase": self.phase,
            "item_pool": self.item_pool.copy(),
            "utilities": self.utilities.copy(),
            "max_turns": self.max_turns,
            "done": self.done,
        }

        # Initialize historical actions with proper shapes
        for i in range(self.nb_agents):
            self.info[f"historical_action_agent_{i}"] = np.zeros(
                (0, self.nb_items), dtype=np.int64
            )

        # Get initial observations
        self.obs = self._get_obs()
        return self.obs, self.info

    def step(self, action):
        """
        Handle one agent's action in either the cheap talk or action phase.
        Modified for vectorized operation.
        """
        if self.next_phase != "None":
            self.phase = self.next_phase
            self.next_phase = "None"
        if self.is_turn_plus_one:
            self.turn += 1
            self.is_turn_plus_one = False
        assert isinstance(
            action, dict
        ), "action should be a dict with 'cheap_talk' and 'proposal' keys"
        assert all(
            key in action for key in ["cheap_talk", "proposal"]
        ), "action must contain both 'cheap_talk' and 'proposal'"

        if self.phase == "cheap_talk":
            self.cheap_talk_proposals[self.waiting_for_agent] = action[
                "cheap_talk"
            ].astype(np.int64)

            if self.waiting_for_agent == self.nb_agents - 1:
                # All agents have made their cheap talk
                self.next_phase = "action"
                self.waiting_for_agent = 0
                reward = 0
                self.done = False
            else:
                self.waiting_for_agent += 1
                reward = 0
                self.done = False

        elif self.phase == "action":
            # Store the action in historical actions with proper shape
            action_proposal = action["proposal"].astype(np.int64)
            current_history = self.info[
                f"historical_action_agent_{self.waiting_for_agent}"
            ]
            if not isinstance(current_history, np.ndarray):
                current_history = np.array(current_history, dtype=np.int64)

            # Ensure proper shape (T, nb_items)
            if len(current_history.shape) == 1:
                current_history = current_history.reshape(-1, self.nb_items)

            # Stack new action
            new_history = (
                np.vstack([current_history, action_proposal.reshape(1, -1)]).astype(
                    np.int64
                )
                if len(current_history) > 0
                else action_proposal.reshape(1, -1).astype(np.int64)
            )

            self.info[f"historical_action_agent_{self.waiting_for_agent}"] = new_history
            self.actual_proposals[self.waiting_for_agent] = action_proposal

            if self.waiting_for_agent == self.nb_agents - 1:
                # Convert proposals to numpy array and check resource constraints
                proposals_array = np.stack(self.actual_proposals)
                sum_proposals = proposals_array.sum(axis=0)

                if np.any(sum_proposals > self.item_pool):
                    reward = 0
                    self.next_phase = "cheap_talk"
                    self.is_turn_plus_one = True

                    # Reset for next round
                    self.waiting_for_agent = 0
                    self.cheap_talk_proposals = [None] * self.nb_agents
                    self.actual_proposals = [None] * self.nb_agents

                    # Check if game should end
                    if self.turn >= self.max_turns:
                        self.done = True
                else:
                    self.next_phase = "reward_calculation"
                    self.waiting_for_agent = 0
                    reward = 0

            else:
                self.waiting_for_agent += 1
                reward = 0
                self.done = False
        elif self.phase == "reward_calculation":
            # Calculate rewards
            if self.waiting_for_agent == 0:
                self.rewards = self._calculate_rewards()
            reward = self.rewards[self.waiting_for_agent]
            if self.waiting_for_agent != self.nb_agents - 1:
                self.waiting_for_agent += 1
                self.done = False
            elif self.waiting_for_agent == self.nb_agents - 1:
                self.is_turn_plus_one = True
                self.done = True
                self.next_phase = "cheap_talk"
                self.waiting_for_agent = 0
                self.cheap_talk_proposals = [None] * self.nb_agents
                self.actual_proposals = [None] * self.nb_agents
        else:
            raise ValueError(f"Invalid phase: {self.phase}")
        # Update info
        self.info.update({"turn": self.turn, "phase": self.phase, "done": self.done})

        # Get new observations
        self.obs = self._get_obs()
        return self.obs, reward, self.done, None, self.info

    def _get_obs(self):
        """Get observation for current agent"""
        # Combine item pool and agent's utilities
        item_context = np.concatenate(
            (self.item_pool, self.utilities[self.waiting_for_agent])
        )

        # Get other agent's cheap talk from previous round
        other_agent = (self.waiting_for_agent + 1) % self.nb_agents
        other_cheap_talk = (
            self.cheap_talk_proposals[other_agent]
            if self.cheap_talk_proposals[other_agent] is not None
            else np.zeros(self.nb_items, dtype=np.int64)
        )

        observations = {
            "item_and_utility": item_context,
            "other_cheap_talk": other_cheap_talk,
        }
        # Add historical actions
        for i in range(self.nb_agents):
            history = self.info[f"historical_action_agent_{i}"]
            if not isinstance(history, np.ndarray):
                history = np.array(history, dtype=np.int64)
            elif history.dtype != np.int64:
                history = history.astype(np.int64)
            # Ensure history maintains shape (T, nb_items) where T is number of steps
            if len(history.shape) == 1:
                history = history.reshape(-1, self.nb_items)
            observations[f"historical_action_agent_{i}"] = history
        return observations

    def _calculate_rewards(self):
        """Calculate rewards based on actual proposals with proper array operations"""
        # Convert proposals to numpy array
        proposals_array = np.stack(
            [prop.astype(np.int64) for prop in self.actual_proposals]
        )

        # Check if sum of proposals exceeds resources
        total_requested = proposals_array.sum(axis=0)
        if np.any(total_requested > self.item_pool):
            return np.array([0, 0], dtype=np.int64)

        # Calculate rewards using dot product
        rewards = np.array(
            [
                np.dot(proposals_array[i], self.utilities[i])
                for i in range(self.nb_agents)
            ],
            dtype=np.int64,
        )

        return rewards


def make_vectorized_env(nb_env, *args, **kwargs):
    """Create multiple environments for parallel training"""

    def make_sim_exchange_env():
        return SimultaneousExchangeEnv(*args, **kwargs)

    envs = gym.vector.SyncVectorEnv([make_sim_exchange_env for _ in range(nb_env)])
    return envs


def test_env():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Test single environment
    env = SimultaneousExchangeEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Item pool:", env.item_pool)
    print("Utilities:", env.utilities)

    # Test a few steps
    # for _ in range(4):  # Two complete rounds (cheap talk + action for each)
    done = False
    while not done:
        # action = {"cheap_talk": np.array([1, 1, 1]), "proposal": np.array([1, 1, 1])}
        action = {
            "cheap_talk": np.random.randint(
                0, env.item_max_quantity + 1, size=env.nb_items
            ),
            "proposal": np.random.randint(
                0, env.item_max_quantity + 1, size=env.nb_items
            ),
        }
        obs, reward, done, _, info = env.step(action)
        print(f"\nStep Results:")
        print("Phase:", info["phase"])
        print("Turn:", info["turn"])
        print("Action:", action)
        print("Reward:", reward)
        print("Done:", done)
        print("Observation:", obs)
    return


def test_vectorized_env():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    nb_env = 4
    nb_items = 3
    item_max_quantity = 5
    nb_agents = 2
    max_turns = {"lower": 4, "mean": 7, "upper": 10}

    # Create vectorized environment
    vec_env = make_vectorized_env(
        nb_env=nb_env,
        nb_items=nb_items,
        item_max_quantity=item_max_quantity,
        nb_agents=nb_agents,
        max_turns=max_turns,
    )

    # Reset all environments
    obs_dict, info = vec_env.reset()

    print("\n=== Initial State ===")
    print(f"Number of environments: {nb_env}")
    for key, value in obs_dict.items():
        if isinstance(value, tuple):
            print(f"{key}:")
            for i, arr in enumerate(value):
                print(f"  env {i} shape: {arr.shape}")
        else:
            print(f"{key} shape: {value.shape}")
    print(f"Initial info: {info}")

    # Run a few steps in all environments simultaneously
    total_steps = 0
    all_done = False

    while not all_done and total_steps < 50:
        total_steps += 1

        # Create batch actions for all environments
        batch_actions = {
            "cheap_talk": np.zeros((nb_env, nb_items), dtype=np.int64),
            "proposal": np.zeros((nb_env, nb_items), dtype=np.int64),
        }

        # Fill with random actions
        for env_idx in range(nb_env):
            batch_actions["cheap_talk"][env_idx] = np.random.randint(
                0, item_max_quantity + 1, size=nb_items
            )
            batch_actions["proposal"][env_idx] = np.random.randint(
                0, item_max_quantity + 1, size=nb_items
            )

        # Step all environments
        obs_dict, rewards, dones, _, infos = vec_env.step(batch_actions)

        print(f"\n=== Step {total_steps} ===")
        print(f"Phase: {infos['phase']}")
        print(f"Turn: {infos['turn']}")
        print(f"Rewards shape: {rewards.shape}")
        print(f"Sample rewards: {rewards}")
        print(f"Dones: {dones}")

        print("\nObservation shapes:")
        for key, value in obs_dict.items():
            # print(f"{key}: {value.shape}")
            if isinstance(value, tuple):
                print(f"{key}:")
                for i, arr in enumerate(value):
                    print(f"  env {i} shape: {arr.shape}")
            else:
                print(f"{key} shape: {value.shape}")

        all_done = all(dones)

    print("\n=== Test Summary ===")
    print(f"Total steps taken: {total_steps}")
    print(f"All environments completed: {all_done}")

    # Close environments
    vec_env.close()


if __name__ == "__main__":
    # Set random seed
    test_env()
    # test_vectorized_env()
