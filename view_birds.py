import time
from stable_baselines3 import PPO
from bird_env import GridBirdsEnv

model = PPO.load("trained_bird_chaser")

# Use render_mode="human" to enable plotting
env = GridBirdsEnv(width=5, height=5, min_birds=1, max_birds=4,
                    max_steps=30, render_mode="human")

num_runs = 10
for _ in range(num_runs):
    obs, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)

env.close()