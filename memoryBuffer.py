from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class memory(object):

    def __init__(self, capacity):
        """Initiates the replay buffer """

        self.memory = deque([], maxlen = capacity) # deque datatype


    def push(self, *args):
        """Saves a transition"""

        self.memory.append(Transition(*args))

    def popleft(self):
        self.memory.popleft()
        return

    def sample(self, batch_size):
        """Samples a batch of size batch_size from the replay buffer"""

        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
        return