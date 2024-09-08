import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from math import isinf

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from observable_operators import p_theta_s0_g_y_stable
from observable_operators import p_theta_s0_g_y
from observable_operators import sample_data
from observable_operators import log_p_theta_obs_g_s0
from observable_operators import log_p_theta_obs
from observable_operators import PRIOR
from observable_operators import sensing_action_sampler
from observable_operators import action_sampler

from finite_state_controller import FSC
from grid_world_1 import Environment

fsc = FSC()
env = Environment()



with open(f'./grid_world_1_data/Values/thetaList_6', "rb") as pkl_rb_obj:
    theta_list = pickle.load(pkl_rb_obj)

iter_num = 1000
M = 100  # number of sampled trajectories
T = 10  # length of a trajectory


def log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta):
    M = len(y_data)
    sum = 0
    for k in range(M):
        log_y_s0 = log_p_theta_obs_g_s0(y_data[k], sa_data[k], s_0, theta)
        log_y = log_p_theta_obs(y_data[k], sa_data[k], s_0, theta)
        if isinf(log_y_s0) and isinf(log_y):
            sum += 0
        else:
            sum += log_y_s0 - log_y
        sum += np.log(PRIOR[s_0, 0])
    return sum


def p_theta_s0_g_multiY(s_0, y_data, sa_data, theta):
    return 2 ** log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta)


def log_prior(s_0, y_data):
    M = len(y_data)
    sum = 0
    for k in range(M):
        sum += np.log(PRIOR[s_0, 0])
    return sum

def sample_data_fixInit(state_0, M, T, theta):
    s_data = np.zeros([M, T], dtype=np.int32)
    a_data = np.zeros([M, T], dtype=np.int32)
    sa_data = []
    y_data = []
    for m in range(M):
        y = []
        sa_list = []
        # start from initial state
        state = state_0
        s = env.states.index(state)
        # Sample sensing action
        me = fsc.memory_space.index('l')
        sAct = sensing_action_sampler(theta, me)
        sa = env.sensing_actions.index(sAct)
        sa_list.append(sa)
        # Get the observation of initial state
        obs = env.observation_function_sampler(state, sAct)
        o = fsc.observations.index(obs)
        me = fsc.transition[me][o]
        y.append(obs)
        # Sample the action from initial state
        act = action_sampler(s)

        for t in range(T):
            s = env.states.index(state)
            s_data[m, t] = s
            a = env.actions.index(act)
            a_data[m, t] = a
            # sample the next state
            state = env.next_state_sampler(state, act)
            # Sample sensing action
            sAct = sensing_action_sampler(theta, me)
            sa = env.sensing_actions.index(sAct)
            sa_list.append(sa)
            # Add the observation
            obs = env.observation_function_sampler(state, sAct)
            o = fsc.observations.index(obs)
            me = fsc.transition[me][o]
            y.append(obs)
            # sample action
            act = action_sampler(s)
        y_data.append(y)
        sa_data.append(sa_list)
    return s_data, a_data, y_data, sa_data

theta = np.random.random([fsc.memory_size, env.sensing_actions_size])
theta[2, 2] = theta[2, 2] + 1000

fix_state_0 = (3, 0)
s_data, a_data, y_data, sa_data = sample_data_fixInit(fix_state_0, M, T, theta)

state_0 = (3, 0)
s_0 = env.states.index(state_0)
print(log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta))

state_0 = (0, 3)
s_0 = env.states.index(state_0)
print(log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta))

state_0 = (5, 2)
s_0 = env.states.index(state_0)
print(log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta))
print('#' * 100)

# print(y_data)

theta = theta_list[-1]

fix_state_0 = (3, 0)
s_data, a_data, y_data, sa_data = sample_data_fixInit(fix_state_0, M, T, theta)

state_0 = (3, 0)
s_0 = env.states.index(state_0)
print(log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta))

state_0 = (0, 3)
s_0 = env.states.index(state_0)
print(log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta))

state_0 = (5, 2)
s_0 = env.states.index(state_0)
print(log_p_theta_s0_g_multiY(s_0, y_data, sa_data, theta))
print('#' * 100)

# print(y_data)

state_0 = (3, 0)
s_0 = env.states.index(state_0)
print(log_prior(s_0, y_data))

state_0 = (0, 3)
s_0 = env.states.index(state_0)
print(log_prior(s_0, y_data))

state_0 = (5, 2)
s_0 = env.states.index(state_0)
print(log_prior(s_0, y_data))
print('#' * 100)
