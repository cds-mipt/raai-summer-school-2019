from gym.envs.registration import register

register(
    id='CarIntersect-v0',
    entry_point='gym_car_intersect.envs:CarRacing', #this is function which you want to code to behave
)
register(
    id='CarIntersect-v1',
    entry_point='gym_car_intersect.envs:CarRacingDiscrete', #this is function which you want to code to behave
)
register(
    id='CarIntersect-v2',
    entry_point='gym_car_intersect.envs:CarRacingHackaton', #this is function which you want to code to behave
)
register(
    id='CarIntersect-v3',
    entry_point='gym_car_intersect.envs:CarRacingHackatonContinuous', #this is function which you want to code to behave
    max_episode_steps=2000,
)

# register(
#     id='CarBehaviour-v0',
#     entry_point='gym_car_intersect.envs:CarBehaviour',
# )
