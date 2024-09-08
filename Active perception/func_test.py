import numpy as np

from grid_world_1 import Environment
# from observable_operators import sample_data
from observable_operators import p_obs_g_sas0
from observable_operators import p_theta_obs

env = Environment()

from finite_state_controller import FSC

fsc = FSC()
# M = 3
# T = 30
# opt_value = env.get_optimal_policy(env.value, env.goals)
# theta = env.extract_opt_theta(env.value, env.goals)
theta = np.ones([fsc.memory_size, env.sensing_actions_size])
# s_data, a_data, y_data, sa_data = sample_data(M, T, theta)
# for i in range(T):
#     print("The state is", env.states[s_data[0][i]])
#     print("The action is", env.actions[a_data[0][i]])
#     print("The observation is", y_data[0][i])
#     print("The sensing action is", env.sensing_actions[sa_data[0][i]])
#     print("#" * 30)
print(p_obs_g_sas0('n33', [2, 2, 2], env.states.index((3, 0))))
print(p_theta_obs('n33', [2, 2, 2], theta))
s = env.states.index((1, 4))
print(env.emission_function(s, '1', '1'))