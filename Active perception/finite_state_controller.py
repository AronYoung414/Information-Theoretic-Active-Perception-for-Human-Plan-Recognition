# Construct a finite memory controller with one-step memory.
import numpy as np
from grid_world_1 import Environment


class FSC:

    def __init__(self):
        env = Environment()
        # Define states
        self.states = env.observations
        # Define initial state
        self.initial_state = 'n'
        # Define actions
        self.actions = env.observations
        # transition probability dictionary
        self.transition = self.get_transition()


    def get_transition(self):
        return 0
