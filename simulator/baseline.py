import os
import gym
import gym_car_intersect

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0

# Create log dir
log_dir = "stable_baseline/saved_models/dqn/"
os.makedirs(log_dir, exist_ok=True)

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

# Create and wrap the environment
from utils import *
env = make_env('CarIntersect-v1', model_name='dqn')

# # Model for participants of hackaton using prepared net
# import torch
# import torch.nn as nn
# net = nn.Sequential(nn.Linear(3*1378**2, 3*4), nn.Sigmoid())
# env = make_env("CarIntersect-v2", net, "torch", model_name='dqn')

# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env.reset()
# Custom MLP policy of two layers of size 512 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")

model = DQN(CustomPolicy, env, verbose=1, tensorboard_log='runs/')
model.learn(total_timesteps=2000000, callback=callback)
# model.save("stable_baseline/dqn_model")

# # remove to demonstrate saving and loading
# del model
# model = DQN.load("stable_baseline/dqn_model")
