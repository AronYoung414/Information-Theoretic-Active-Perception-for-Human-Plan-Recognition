import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from math import isinf

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from observable_operators import p_theta_obs_g_s0
from observable_operators import p_theta_s0_g_y_stable
from observable_operators import log_p_theta_obs_g_s0
from observable_operators import log_p_theta_obs
from observable_operators import PRIOR
from observable_operators import sensing_action_sampler
from observable_operators import action_sampler

from finite_state_controller import FSC
from grid_world_2_obstacle import Environment

fsc = FSC()
env = Environment()


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


def p_theta_s0_g_multiY_stable(y_data, sa_data, theta, scale_factor=1e10):
    prob_list = np.ones(len(env.initial_states))
    M = len(y_data)
    for k in range(M):
        y_k = y_data[k]
        sa_list_k = sa_data[k]
        for i in range(len(env.initial_states)):
            state_0 = env.initial_states[i]
            s_0 = env.states.index(state_0)
            p_theta_y_s0 = p_theta_obs_g_s0(y_k, sa_list_k, s_0, theta)
            prob_list[i] *= p_theta_y_s0 * scale_factor
    for i in range(len(env.initial_states)):
        state_0 = env.initial_states[i]
        s_0 = env.states.index(state_0)
        prob_list[i] *= PRIOR[s_0, 0]
    total = sum(prob_list)
    if total == 0:
        normalized_list = [1 / len(prob_list) for i in range(len(prob_list))]
    else:
        normalized_list = [x / total for x in prob_list]
    return normalized_list


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


# Correct_thetaList_1 is the results of new experiment
with open(f'grid_world_2_data/Values/Correct_thetaList_1', "rb") as pkl_rb_obj:
    theta_list = pickle.load(pkl_rb_obj)

theta_r = np.random.random([fsc.memory_size, env.sensing_actions_size])
theta = theta_list[-1]

iter_num = 1000
M = 1  # number of sampled trajectories
T = 10  # length of a trajectory

for type_num in range(1, 4):
    fix_state_0 = env.initial_states[type_num - 1]

    s_data, a_data, y_data, sa_data = sample_data_fixInit(fix_state_0, M, T, theta_r)
    y_list = y_data[0]
    sa_list = sa_data[0]
    type_1_list_r = []
    type_2_list_r = []
    type_3_list_r = []
    for t in range(T):
        type_1_list_r.append(p_theta_s0_g_y_stable(y_list[:t + 1], sa_list[:t + 1], theta_r)[0])
        type_2_list_r.append(p_theta_s0_g_y_stable(y_list[:t + 1], sa_list[:t + 1], theta_r)[1])
        type_3_list_r.append(p_theta_s0_g_y_stable(y_list[:t + 1], sa_list[:t + 1], theta_r)[2])

    s_data, a_data, y_data, sa_data = sample_data_fixInit(fix_state_0, M, T, theta)
    print([env.sensing_actions[sa] for sa in sa_data[0]])
    y_list = y_data[0]
    sa_list = sa_data[0]
    type_1_list = []
    type_2_list = []
    type_3_list = []
    for t in range(T):
        type_1_list.append(p_theta_s0_g_y_stable(y_list[:t + 1], sa_list[:t + 1], theta)[0])
        type_2_list.append(p_theta_s0_g_y_stable(y_list[:t + 1], sa_list[:t + 1], theta)[1])
        type_3_list.append(p_theta_s0_g_y_stable(y_list[:t + 1], sa_list[:t + 1], theta)[2])

    # with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_1_list_r_{type_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(type_1_list_r, pkl_wb_obj)
    #
    # with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_2_list_r_{type_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(type_2_list_r, pkl_wb_obj)
    #
    # with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_3_list_r_{type_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(type_3_list_r, pkl_wb_obj)
    #
    # with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_1_list_{type_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(type_1_list, pkl_wb_obj)
    #
    # with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_2_list_{type_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(type_2_list, pkl_wb_obj)
    #
    # with open(f'./grid_world_2_data/Values/Results_Analysis_Fixed/type_3_list_{type_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(type_3_list, pkl_wb_obj)

    # Create plot
    fig, ax = plt.subplots()
    # Plot lines
    line1, = ax.plot(range(T), type_1_list_r, ':b.', label='type 1 (random)')
    line2, = ax.plot(range(T), type_2_list_r, ':rD', label='type 2 (random)')
    line3, = ax.plot(range(T), type_3_list_r, ':gs', label='type 3 (random)')
    line4, = ax.plot(range(T), type_1_list, '-b.', label='type 1 (min_entropy)')
    line5, = ax.plot(range(T), type_2_list, '-rD', label='type 2 (min_entropy)')
    line6, = ax.plot(range(T), type_3_list, '-gs', label='type 3 (min_entropy)')
    plt.xlabel(r"The time step $t$")
    plt.ylabel("The belief (posterior initial distribution)")
    plt.title(f"The Evolution of Belief (True Type is {type_num})")
    # Create first legend for the first 3 lines
    first_legend = ax.legend([line1, line2, line3],
                             ['type 1 (random)', 'type 2 (random)', 'type 3 (random)'],
                             loc='upper left', bbox_to_anchor=(0, 1))
    # Add the first legend manually to the plot
    ax.add_artist(first_legend)
    # Create a second legend for other lines
    ax.legend([line4, line5, line6],
              ['type 1 (min_entropy)', 'type 2 (min_entropy)', 'type 3 (min_entropy)'],
              loc='lower right', bbox_to_anchor=(1, 0.3))
    # plt.savefig(f'./grid_world_2_data/Graphs/Results_Analysis_Fixed/type_{type_num}.png')
    # plt.savefig(f'./grid_world_2_data/Graphs/Results_Analysis_Fixed/type_{type_num}.pdf', format="pdf",
    #             bbox_inches="tight")
    plt.show()
