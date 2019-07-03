import gym
import gym_car_intersect
import matplotlib.pyplot as plt

env = gym.make("CarIntersect-v2")
plt.imshow(env.reset())

import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(3*1378**2, 3*4), nn.Sigmoid())

from utils import *
env = make_env("CarIntersect-v2", net, "torch")
print(env.reset())


# plt.imshow(env.reset())
#
# for _ in range(1):
#     s, r, done, _ = env.step(env.action_space.sample())
#     plt.imshow(s)
#     plt.show()
