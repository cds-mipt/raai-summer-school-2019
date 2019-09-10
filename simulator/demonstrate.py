import gym
import numpy as np


from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

# Create and wrap the environment
from env_wrappers import make_pixel_env
env = make_pixel_env('CarIntersect-v3')

model = SAC(MlpPolicy, env, verbose=1)
model = SAC.load("sac_car-v3")


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

done = False
obs = env.reset()
obj = ax.imshow(s[1])
while not done:
    action, _states = model.predict(obs)
    obs, r, done, _ = env.step(action)
    obj.set_data(obs[1])
    fig.canvas.draw()
    plt.imshow(s[1])
    plt.show()

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
