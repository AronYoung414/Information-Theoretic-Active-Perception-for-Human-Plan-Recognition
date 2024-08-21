from observable_operators import get_memory_space
from observable_operators import log_policy_gradient
from observable_operators import p_obs_g_sas0
from observable_operators import p_obs_g_sas0_initial
from observable_operators import observable_operator
import numpy as np

from grid_world_1 import Environment

env = Environment()
Tmat = env.transition_wc
K = 1
# print(get_memory_space(K)[0])
# print(log_policy_gradient(K, 2, 2, theta=np.ones([env.observations_size ** K, env.action_size])))
ob = observable_operator('3', '3')
print(np.linalg.norm(observable_operator('3', '3')))
print(p_obs_g_sas0('n33', ['3', '3', '3'], env.states.index((3, 0))))  # There may be some problems with this function
print(p_obs_g_sas0_initial('n', '3', env.states.index((3, 0))))
