import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import os

algorithm = "A2C"
model_dir = "models/" + algorithm
logs_dir = "logs"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = make_vec_env("LunarLander-v2", n_envs=1)
obs, info = env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS = int(10e3)
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=algorithm)
    model.save(f"{model_dir}/{TIMESTEPS*i}")

'''
episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)'''
env.close()