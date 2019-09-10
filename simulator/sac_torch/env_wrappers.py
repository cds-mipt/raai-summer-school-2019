import time
import copy
import math
import datetime
from collections import OrderedDict
from logging import getLogger

import cv2
import gym
from gym import spaces
import numpy as np
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder

import pickle
import cloudpickle
from multiprocessing import Process, Pipe, Pool

import gym_car_intersect
from tqdm import trange
import collections


cv2.ocl.setUseOpenCL(False)
logger = getLogger(__name__)


class ResetTrimInfoWrapper(gym.Wrapper):
    """Take first return value.
    minerl's `env.reset()` returns tuple of `(obs, info)`
    but existing agent implementations expect `reset()` returns `obs` only.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs


class ContinuingTimeLimitMonitor(Monitor):
    """`Monitor` with ChainerRL's `ContinuingTimeLimit` support.
    Because of the original implementation's design,
    explicit `close()` is needed to save the last episode.
    Do not forget to call `close()` at the last line of your script.
    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """

    def _start(self, directory, video_callable=None, force=False, resume=False,
               write_upon_reset=False, uid=None, mode=None):
        if self.env_semantics_autoreset:
            raise gym.error.Error(
                "Detect 'semantics.autoreset=True' in `env.metadata`, "
                "which means the env comes from deprecated OpenAI Universe.")
        ret = super()._start(directory=directory,
                             video_callable=video_callable, force=force,
                             resume=resume, write_upon_reset=write_upon_reset,
                             uid=uid, mode=mode)
        if self.env.spec is None:
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id
        self.stats_recorder = _ContinuingTimeLimitStatsRecorder(
            directory,
            '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix),
            autoreset=False, env_id=env_id)
        return ret


class _ContinuingTimeLimitStatsRecorder(StatsRecorder):
    """`StatsRecorder` with ChainerRL's `ContinuingTimeLimit` support.
    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/stats_recorder.py
    """

    def __init__(self, directory, file_prefix, autoreset=False, env_id=None):
        super().__init__(directory, file_prefix,
                         autoreset=autoreset, env_id=env_id)
        self._save_completed = True

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            logger.debug('Tried to reset env which is not done. '
                         'StatsRecorder completes the last episode.')
            self.save_complete()

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_step(self, observation, reward, done, info):
        self._save_completed = False
        return super().after_step(observation, reward, done, info)

    def save_complete(self):
        if not self._save_completed:
            super().save_complete()
            self._save_completed = True

    def close(self):
        self.save_complete()
        super().close()


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov']


class ObtainObservationWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env, observations_key):
        super().__init__(env)

        self.observations_key = observations_key
        new_observation_space = {}

        for key in observations_key:
            assert key in self.observation_space.spaces, f'{key} is not valid for current environment'
            if key == 'pov':
                new_observation_space['pov'] = self.env.observation_space.spaces['pov']
            if key == 'compassAngle':
                new_observation_space['compassAngle'] = self.env.observation_space.spaces['compassAngle']
            if key == 'equipped_items':
                low_array = np.zeros(8)
                high_array = np.ones(8)
                new_observation_space['equipped_items'] = spaces.Box(low_array, high_array, dtype=np.float32)
            if key == 'inventory':
                inventory_size = len(self.env.observation_space.spaces['inventory'].spaces)
                low_array = np.zeros(inventory_size)
                high_array = float('inf')*np.ones(inventory_size)
                new_observation_space['inventory'] = spaces.Box(low_array, high_array, dtype=np.float32)

        self.observation_space = new_observation_space

    def observation(self, observation):
        new_observation = {}
        for key in self.observation_space:
            if key == 'pov':
                new_observation['pov'] = observation['pov']
            if key == 'compassAngle':
                new_observation['compassAngle'] = observation['compassAngle']
            if key == 'equipped_items':
                array = [0]*8 # as there is 8 different items in equip, and this is ohe
                array[observation['equipped_items']['mainhand']['type']] = 1
                new_observation['equipped_items'] = array
            if key == 'inventory':
                new_observation['inventory'] = [v for k, v in observation['inventory'].items()]

        return new_observation


class PoVWithInventoryWrapper(gym.ObservationWrapper):
    """Take 'pov' value (current game display) and concatenate inventory information with it, as a new channel of image;
    resulting image has RGB+inventory (or K+compass for gray-scaled image) channels.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space['pov']

    def observation(self, observation):
        pov = observation['pov']
        inventory_scaled = np.array(observation['inventory'], dtype=pov.dtype)  / 255.0
        size = math.ceil(pov.shape[0]/inventory_scaled.shape[0])
        inventory_channel = np.tile(inventory_scaled, (pov.shape[0], size))[:, :pov.shape[1], None]
        return np.concatenate([pov, inventory_channel], axis=-1)


class PoVWithCompassAngleWrapper(gym.ObservationWrapper):
    """Take 'pov' value (current game display) and concatenate compass angle information with it, as a new channel of image;
    resulting image has RGB+compass (or K+compass for gray-scaled image) channels.
    """
    def __init__(self, env):
        super().__init__(env)

        self._compass_angle_scale = 180 / 255  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later

        pov_space = self.env.observation_space.spaces['pov']
        compass_angle_space = self.env.observation_space.spaces['compassAngle']

        low = self.observation({'pov': pov_space.low, 'compassAngle': compass_angle_space.low})
        high = self.observation({'pov': pov_space.high, 'compassAngle': compass_angle_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, observation):
        pov = observation['pov']
        compass_scaled = observation['compassAngle'] / self._compass_angle_scale
        compass_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=pov.dtype) * compass_scaled
        return np.concatenate([pov, compass_channel], axis=-1)


class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""
    def __init__(self, env, source, destination):
        assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)

        self.source = source
        self.destination = destination

        low = self.observation(self.observation_space.low)
        high = self.observation(self.observation_space.high)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def observation(self, frame):
        return np.moveaxis(frame, self.source, self.destination)


class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)

        self._key = dict_space_key
        # print("Env:\n", env.observation_space)
        # print("Self:\n", self.observation_space)

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]
        height, width = original_space.shape[0], original_space.shape[1]

        # # sanity checks
        # ideal_image_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        # if original_space != ideal_image_space:
        #     raise ValueError('Image space should be {}, but given {}.'.format(ideal_image_space, original_space))
        # if original_space.dtype != np.uint8:
        #     raise ValueError('Image should `np.uint8` typed, but given {}.'.format(original_space.dtype))

        height, width = original_space.shape[0], original_space.shape[1]
        new_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        if self._key is None:
            self.observation_space = new_space
        else:
            self.observation_space.spaces[self._key] = new_space

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        if self._key is None:
            obs = frame
        else:
            obs[self._key] = frame
        return obs


class SerialDiscreteActionWrapper(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.
    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.
    Parameters
    ----------
    env
        Wrapping gym environment.
    always_keys
        List of action keys, which should be always pressed throughout interaction with environment.
        If specified, the "noop" action is also affected.
    reverse_keys
        List of action keys, which should be always pressed but can be turn off via action.
        If specified, the "noop" action is also affected.
    exclude_keys
        List of action keys, which should be ignored for discretizing action space.
    exclude_noop
        The "noop" will be excluded from discrete action list.
    """

    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, always_keys=None, reverse_keys=None, exclude_keys=None, exclude_noop=False):
        super().__init__(env)

        self.always_keys = [] if always_keys is None else always_keys
        self.reverse_keys = [] if reverse_keys is None else reverse_keys
        self.exclude_keys = [] if exclude_keys is None else exclude_keys
        if len(set(self.always_keys) | set(self.reverse_keys) | set(self.exclude_keys)) != \
                len(self.always_keys) + len(self.reverse_keys) + len(self.exclude_keys):
            raise ValueError('always_keys ({}) or reverse_keys ({}) or exclude_keys ({}) intersect each other.'.format(
                self.always_keys, self.reverse_keys, self.exclude_keys))
        self.exclude_noop = exclude_noop

        self.wrapping_action_space = self.env.action_space
        self._noop_template = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack' , 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
        ])
        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self._noop_template:
                raise ValueError('Unknown action name: {}'.format(key))

        # get noop
        self.noop = copy.deepcopy(self._noop_template)
        for key in self._noop_template:
            if key not in self.wrapping_action_space.spaces:
                del self.noop[key]

        # check&set always_keys
        for key in self.always_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `always_keys`.'.format(key))
            self.noop[key] = 1
        logger.info('always pressing keys: {}'.format(self.always_keys))
        # check&set reverse_keys
        for key in self.reverse_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `reverse_keys`.'.format(key))
            self.noop[key] = 1
        logger.info('reversed pressing keys: {}'.format(self.reverse_keys))
        # check exclude_keys
        for key in self.exclude_keys:
            if key not in self.noop:
                raise ValueError('unknown exclude_keys: {}'.format(key))
        logger.info('always ignored keys: {}'.format(self.exclude_keys))

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key in self.always_keys or key in self.exclude_keys:
                continue
            if key in self.BINARY_KEYS:
                # action candidate : {1}  (0 is ignored because it is for noop), or {0} when `reverse_keys`.
                op = copy.deepcopy(self.noop)
                if key in self.reverse_keys:
                    op[key] = 0
                else:
                    op[key] = 1
                self._actions.append(op)
            elif key == 'camera':
                # action candidate : {[0, -10], [0, 10]}
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
            elif key in {'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'}:
                # action candidate : {1, 2, ..., len(space)-1}  (0 is ignored because it is for noop)
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)
        if self.exclude_noop:
            del self._actions[0]

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


class CombineActionWrapper(gym.ActionWrapper):
    """Combine MineRL env's "exclusive" actions.
    "exclusive" actions will be combined as:
        - "forward", "back" -> noop/forward/back (Discrete(3))
        - "left", "right" -> noop/left/right (Discrete(3))
        - "sneak", "sprint" -> noop/sneak/sprint (Discrete(3))
        - "attack", "place", "equip", "craft", "nearbyCraft", "nearbySmelt"
            -> noop/attack/place/equip/craft/nearbyCraft/nearbySmelt (Discrete(n))
    The combined action's names will be concatenation of originals, i.e.,
    "forward_back", "left_right", "snaek_sprint", "attack_place_equip_craft_nearbyCraft_nearbySmelt".
    """
    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        def combine_exclusive_actions(keys):
            """
            Dict({'forward': Discrete(2), 'back': Discrete(2)})
            =>
            new_actions: [{'forward':0, 'back':0}, {'forward':1, 'back':0}, {'forward':0, 'back':1}]
            """
            new_key = '_'.join(keys)
            valid_action_keys = [k for k in keys if k in self.wrapping_action_space.spaces]
            noop = {a: 0 for a in valid_action_keys}
            new_actions = [noop]

            for key in valid_action_keys:
                space = self.wrapping_action_space.spaces[key]
                for i in range(1, space.n):
                    op = copy.deepcopy(noop)
                    op[key] = i
                    new_actions.append(op)
            return new_key, new_actions

        self._maps = {}
        for keys in (
                ('forward', 'back'), ('left', 'right'), ('sneak', 'sprint'),
                ('attack', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt')):
            new_key, new_actions = combine_exclusive_actions(keys)
            self._maps[new_key] = new_actions

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        self.action_space = gym.spaces.Dict({
            'forward_back':
                gym.spaces.Discrete(len(self._maps['forward_back'])),
            'left_right':
                gym.spaces.Discrete(len(self._maps['left_right'])),
            'jump':
                self.wrapping_action_space.spaces['jump'],
            'sneak_sprint':
                gym.spaces.Discrete(len(self._maps['sneak_sprint'])),
            'camera':
                self.wrapping_action_space.spaces['camera'],
            'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                gym.spaces.Discrete(len(self._maps['attack_place_equip_craft_nearbyCraft_nearbySmelt']))
        })

        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))
        for k, v in self._maps.items():
            logger.info('{} -> {}'.format(k, v))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = OrderedDict()
        for k, v in action.items():
            if k in self._maps:
                a = self._maps[k][v]
                original_space_action.update(a)
            else:
                original_space_action[k] = v

        logger.debug('action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


class SerialDiscreteCombineActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key == 'camera':
                # action candidate : {[0, -10], [0, 10]}
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
            else:
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


# class ImageToPyTorch(gym.ObservationWrapper):
#     def __init__(self, env):
#         super(ImageToPyTorch, self).__init__(env)
#         assert 'pov' in self.observation_space, 'Wrapper ImageToPyTorch is not applicable withot pov.'
#         old_shape = self.observation_space['pov'].shape
#         self.observation_space['pov'] = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
#                                                 dtype=np.float32)
#
#     def observation(self, observation):
#         observation['pov'] = np.moveaxis(observation['pov'], 2, 0)
#         return observation


# class ScaledFloatFrame(gym.ObservationWrapper):
#     def observation(self, obs):
#         obs['pov'] = np.array(obs['pov']).astype(np.float32) / 255.0
#         return obs


class ContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, env, actions_key):
        super().__init__(env)
        action_shape = len(actions_key)+1 if 'camera' in actions_key else len(actions_key)
        if action_shape == len(actions_key)+1:
            assert actions_key[-1] == 'camera', 'Make sure that "camera" key is the last element among actions_key'
        low_array = -np.ones(action_shape)
        high_array = np.ones(action_shape)
        self.all_actions_key = list(self.action_space.spaces.keys())
        self.action_space = spaces.Box(low_array, high_array, dtype=np.float32)
        self.actions_key = {key: idx
                            for idx, key in enumerate(actions_key)}

    def action(self, action):
        dict_action = OrderedDict()
        for key in self.all_actions_key:
            if key in self.actions_key:
                i = self.actions_key[key]
                if key == 'camera':
                    dict_action[key] = [5*action[-2], 5*action[-1]]
                elif key == 'craft':
                    bins = np.linspace(-1, 1, 5)
                    dict_action[key] = np.digitize(action[i], bins)-1
                elif key == 'equip':
                    bins = np.linspace(-1, 1, 8)
                    dict_action[key] = np.digitize(action[i], bins)-1
                elif key == 'nearbyCraft':
                    bins = np.linspace(-1, 1, 8)
                    dict_action[key] = np.digitize(action[i], bins)-1
                elif key == 'nearbySmelt':
                    bins = np.linspace(-1, 1, 3)
                    dict_action[key] = np.digitize(action[i], bins)-1
                elif key == 'place':
                    bins = np.linspace(-1, 1, 6)
                    dict_action[key] = np.digitize(action[i], bins)-1
                else:
                    dict_action[key] = 1 if action[i] > 0 else 0
            else:
                if key == 'camera':
                    dict_action[key] = [0, 0]
                else:
                    dict_action[key] = 0
        return dict_action


class Summaries(gym.Wrapper):
    """ Wrapper to write summaries. """
    number_of_episodes = 0
    writer = None

    def step(self, action):
        output = self.env.step(action)
        s, r, done, _ = output
        self.total_reward += r
        if done:
            Summaries.number_of_episodes += 1
            if Summaries.writer is not None:
                Summaries.writer.add_scalar("episode reward",
                                        self.total_reward,
                                        Summaries.number_of_episodes)
        return output

    def reset(self, **kwargs):
        self.total_reward = 0
        return self.env.reset(**kwargs)


# Custom function for multi env:
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv():
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        no_of_envs = len(env_fns)
        self.remotes, self.work_remotes = \
            zip(*[Pipe() for _ in range(no_of_envs)])
        self.ps = []

        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process(target = worker, args = (wrk, rem, CloudpickleWrapper(fn)))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = False
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        self.observation_space, self.action_space, self.spec = self.remotes[0].recv()

    def step_async(self, actions):
        if self.waiting:
            raise AlreadySteppingError
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        if not self.waiting:
            raise NotSteppingError
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        if isinstance(obs[0], OrderedDict):
            obs_dict = OrderedDict({k : np.stack([d[k] for d in obs]) for k in obs[0].keys()})
        else:
            obs_dict = np.stack(obs)
        return obs_dict, np.stack(rews), np.stack(dones), np.stack(infos)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        if isinstance(obs[0], OrderedDict):
            obs_dict = OrderedDict({k : np.stack([d[k] for d in obs]) for k in obs[0].keys()})
        else:
            obs_dict = np.stack(obs)
        return obs_dict

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# Just copied environments:

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(150, 150, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        img = np.reshape(frame, [1378, 1378, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        x_t = resized_screen#[18:102, :]
        x_t = np.reshape(x_t, [150, 150, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class Image2Coord(gym.ObservationWrapper):
    def __init__(self, env):
        self.observation_space = 

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


def make_env(name='CartPole-v0'):
    env = gym.make(name)
    return env


def make_pixel_env(name='CartPole-v0'):
    env = gym.make(name)
    env = MaxAndSkipEnv(env)
    # env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env


# def make_env(env_name, observations_key, actions_key):
#     """ Combine several wrappers around env. """
#     if env_name == 'MineRLSimple-v0':
#         import SimpleEnvironment
#     env = gym.make(env_name)
#     # env = ResetTrimInfoWrapper(env)
#     env = GrayScaleWrapper(env, dict_space_key='pov')
#     env = ObtainObservationWrapper(env, observations_key)
#     env = ScaledFloatFrame(env)
#     env = PoVWithInventoryWrapper(env)
#     # env = ImageToPyTorch(env)
#     env = ContinuousActionWrapper(env, actions_key)
#     return env
