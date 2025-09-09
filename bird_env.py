import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridBirdsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        width:      int = 10,
        height:     int = 10,
        min_birds:  int = 1,
        max_birds:  int = 5,
        max_steps:  int = 100,
        sound_range:float = 1.5,
        render_mode:str | None = None,
    ):
        super().__init__()

        assert 1 <= min_birds <= max_birds

        # World size
        self.width = width
        self.height = height

        # Agent position
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        # Bird population range
        self.min_birds = min_birds
        self.max_birds = max_birds
        # Actual number of birds
        self.num_birds = 0
        # Bird 2D location matrix, -1 indicating no location
        self.bird_pos = np.full((self.max_birds, 2), -1, dtype=np.int32)
        # Distance from which birds can be heard
        self.sound_range = sound_range

        # Action Space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([ 1,  1, self.max_birds], dtype=np.float32),
            dtype=np.float32
        )
        # Observations: Agent X & Y, each bird's volume
        obs_size = 2 + self.max_birds
        self.observation_space = spaces.Box(
            low=-1.0, high=max(self.width, self.height),
            shape=(obs_size,), dtype=np.float32
        )

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
            
        centroid = np.mean(self.bird_pos[: self.num_birds], axis=0)
        self.prev_dist = np.linalg.norm(self.agent_pos - centroid)

        return self._get_obs(), {}

    def _get_obs(self):
        # Bird sounds (volumes = inverse distance)
        sounds = []
        for bird in self.bird_pos:
            d = np.linalg.norm(self.agent_pos - bird)
            if d <= self.sound_range:
                sounds.append(1.0 / (d + 1e-3))
            else:
                sounds.append(0.0)

        while len(sounds) < self.max_birds:
            sounds.append(0.0)

        return np.concatenate([self.agent_pos, sounds], dtype=np.float32)
    
    def step(self, actions):
        self.step_count += 1

        agent_move = actions[:2]
        agent_guess = int(np.round(actions[2]))
        self.agent_pos = np.clip(self.agent_pos + agent_move,
                                 0, [self.width, self.height])

        reward = 0.0

        # Goal: reach centroid of birds
        centroid = np.mean(self.bird_pos[: self.num_birds], axis=0)
        dist = np.linalg.norm(self.agent_pos - centroid)

        # Reward: positive if closer, negative if farther
        reward = self.prev_dist - dist
        self.prev_dist = dist

        # Bonus for reaching centroid and terminate
        if dist < 0.5:
            reward += 10.0
            terminated = True
        else:
            terminated = False

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
        for bird in self.bird_pos:
            plt.scatter(bird[0], bird[1], c="red", marker="x")
            circle = patches.Circle(bird, radius=self.sound_range,
                                    fill=False, color="gray", linestyle="--",
                                    alpha=0.5)
            plt.gca().add_patch(circle)

        # Pause for humans
        plt.pause(0.01)

    def close(self):
        plt.close()

import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = GridBirdsEnv(width=5, height=5, min_birds=1, max_birds=4,
                       max_steps=30)
    obs, info = env.reset()

    # Define model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train for some timesteps
    model.learn(total_timesteps=50_000)

    # Save model
    model.save("trained_bird_chaser")

    # Use render_mode="human" to enable plotting
    env = GridBirdsEnv(width=5, height=5, min_birds=1, max_birds=4,
                       max_steps=30, render_mode="human")
    obs, info = env.reset()

    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)

    env.close()
