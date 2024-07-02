# Construct a big HMM with two branches. Each branch represents a type of agent.
from grid_world_1 import Environment
import numpy as np
GAMMA = 0.9
env = Environment()





class HMM:

    def __init__(self):
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
