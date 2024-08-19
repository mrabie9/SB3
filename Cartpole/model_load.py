import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import os

algorithm = "PPO-2"
model_dir = "models/" + algorithm
logs_dir = "logs"

vec_env = make_vec_env("CartPole-v1", n_envs=4)
vec_env.reset()

model_path = f"{model_dir}/80000"
model = PPO.load(model_path)

episodes = 10

for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    t = time.time()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
    print("Done", time.time()-t)
vec_env.close()