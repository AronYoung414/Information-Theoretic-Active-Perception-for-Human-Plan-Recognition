import numpy as np
from observable_operators import get_memory_space


class FSC:

    def __init__(self):
        # The length of memory
        self.K = 2
        # the memory space
        self.state_space = get_memory_space(self.K)
