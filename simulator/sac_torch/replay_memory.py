import random
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if isinstance(done, np.ndarray):
            self.buffer.extend(list(zip(state, action, reward, next_state, done)))
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class OptionReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done, option):
        if isinstance(done, np.ndarray):
            self.buffer.extend(list(zip(state, action, reward, next_state, done, option)))
        else:
            self.buffer.append((state, action, reward, next_state, done, option))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, option = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, option

    def __len__(self):
        return len(self.buffer)
