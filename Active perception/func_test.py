from observable_operators import get_memory_space
from observable_operators import log_policy_gradient
from observable_operators import p_obs_g_sas0
import numpy as np

from grid_world_1 import Environment
env = Environment()

K = 1
print(get_memory_space(K)[0])
print(log_policy_gradient(K, 2, 2, theta = np.ones([env.observations_size ** K, env.action_size])))
print(p_obs_g_sas0('nnn', ['1', '2', '3'], env.states.index((3, 0)))) # There may be some problems with this function