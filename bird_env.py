import random
from typing import Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class GridBirdsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        width:      int = 10,
        height:     int = 10,
        min_birds:  int = 1,
        max_birds:  int = 5,
        max_steps:  int = 100,
        seed:       int | None = None,
        render_mode:str | None = None,
    ):
        super().__init__()

        assert 1 <= min_birds <= max_birds

        # World size
        self.width = width
        self.height = height

        # Agent position
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Dict()

        # Bird population range
        self.min_birds = min_birds
        self.max_birds = max_birds
        # Actual number of birds
        self.num_birds = 0
        # Bird 2D location matrix, -1 indicating no location
        self.bird_pos = np.full((self.max_birds, 2), -1, dtype=np.int32)

        # Steps per episode
        self.max_steps = max_steps
        # Current step
        self.step_count = 0

        # Render using matplotlib
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)

        self.step_count = 0
        self.agent_pos = [int(rng.integers(0, self.width)),
             int(rng.integers(0, self.height))]
        self.num_birds = int(rng.integers(self.min_birds, self.max_birds + 1))

        self.bird_pos[:] = -1
        for i in range(self.num_birds):
            self.bird_pos[i] = [int(rng.integers(0, self.width)),
             int(rng.integers(0, self.height))]

        return self._get_obs(), {}

    def _get_obs(self) -> Dict:
        return {
            "agent": self.agent_pos.copy(),
            "birds": self.bird_pos.copy(),
            "num_birds": int(self.num_birds),
        }

    def step(self, actions):
        self.step_count += 1

        self.agent_pos = np.clip(self.agent_pos + actions,
                                 0, [self.width, self.height])

        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self):
        # Initialise plot
        plt.clf()
        plt.xlim(-0.5, self.width + 0.5)
        plt.ylim(-0.5, self.height + 0.5)
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.title(f"Step {self.step_count}")

        # Draw agents and birds
        plt.scatter(self.agent_pos[0], self.agent_pos[1], c="red", s=200,
                    marker="o", label="Agent")
        plt.scatter(self.bird_pos[:,0], self.bird_pos[:,1], c="blue", s=200,
                    marker="x", label="Birds")

        # Pause for humans
        plt.pause(0.01)

    def close(self):
        plt.close()

if __name__ == "__main__":
    env = GridBirdsEnv(width=5, height=5, min_birds=1, max_birds=4,
                       max_steps=10, render_mode="human")
    obs, info = env.reset()

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

    env.close()