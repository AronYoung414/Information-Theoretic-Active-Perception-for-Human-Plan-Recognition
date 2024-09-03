import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from random import choices, choice
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from grid_world_1 import Environment
from finite_state_controller import FSC

env = Environment()
fsc = FSC()


def pi_theta(m, sa, theta):
    """
    :param m: the index of a finite sequence of observation, corresponding to K-step memory
    :param sa: the sensing action to be given
    :param theta: the policy parameter, the memory_size * sensing_action_size
    :return: the Gibbs policy given the finite memory
    """
    e_x = np.exp(theta[m, :] - np.max(theta[m, :]))
    return (e_x / e_x.sum(axis=0))[sa]


def log_policy_gradient(m, sa, theta):
    # A memory space for K-step memory policy
    memory_space = fsc.memory_space
    memory_size = fsc.memory_size
    gradient = np.zeros([memory_size, env.sensing_actions_size])
    memory = memory_space[m]
    sen_act = env.sensing_actions[sa]
    for m_prime in range(memory_size):
        for a_prime in range(env.sensing_actions_size):
            memory_p = memory_space[m_prime]
            senact_p = env.sensing_actions[a_prime]
            indicator_m = 0
            indicator_a = 0
            if memory == memory_p:
                indicator_m = 1
            if sen_act == senact_p:
                indicator_a = 1
            partial_pi_theta = indicator_m * (indicator_a - pi_theta(m_prime, a_prime, theta))
            gradient[m_prime, a_prime] = partial_pi_theta
    return gradient


def observable_operator(o_t, a_t):
    oo = np.zeros([env.state_size, env.state_size])
    for i in range(env.state_size):
        for j in range(env.state_size):
            # The definition of observable operators
            oo[i, j] = env.transition_wc[j, i] * env.emission_function(j, a_t, o_t)
    return oo


def p_obs_g_sas0(y, senAct_list, s_0):
    # Give value to the initial state
    mu_0 = np.zeros([env.state_size, 1])
    mu_0[s_0, 0] = 1
    # Obtain observable operators
    oo = observable_operator(y[-1], senAct_list[-1])
    # Creat a vector with all elements equals to 1
    one_vec = np.ones([1, env.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    # Calculate the probability
    for t in reversed(range(len(y) - 1)):
        oo = observable_operator(y[t], senAct_list[t])
        probs = probs @ oo
    probs_1 = probs @ mu_0
    probs_2 = probs @ env.initial_state_dis
    return probs_1[0][0], probs_2[0][0]


def p_obs_g_sas0_initial(o_0, senAct_0, s_0):
    mu_0 = np.zeros([env.state_size, 1])
    mu_0[s_0, 0] = 1
    # Obtain observable operators
    oo = observable_operator(o_0, senAct_0)
    # Creat a vector with all elements equals to 1
    one_vec = np.ones([1, env.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    probs_1 = probs @ mu_0
    probs_2 = probs @ env.initial_state_dis
    return probs_1[0][0], probs_2[0][0]


def p_theta_obs_g_s0(y, sa_list, s_0, theta):
    """
    :param y: the sequence of observations given states and sensing actions
    :param sa_list: the sequence of sensing actions
    :param s_0: the initial state
    :param theta: the sensing policy parameter
    :return: the probability P(y, sa_list| s_0 ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_prod = pi_theta(m, sa_list[0], theta)
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_prod *= pi_theta(m, sa_list[i + 1], theta)
    return p_obs_g_sas0(y, sa_list, s_0)[0] / p_obs_g_sas0_initial(y[0], sa_list[0], s_0)[0] * policy_prod


def log_p_theta_obs_g_s0(y, sa_list, s_0, theta):
    """
    :param y: the sequence of observations given states and sensing actions
    :param sa_list: the sequence of sensing actions
    :param s_0: the initial state
    :param theta: the sensing policy parameter
    :return: the log probability log P(y, sa_list| s_0 ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_sum = np.log2(pi_theta(m, sa_list[0], theta))
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_sum += np.log2(pi_theta(m, sa_list[i + 1], theta))
    return np.log2(p_obs_g_sas0(y, sa_list, s_0)[0]) - np.log2(
        p_obs_g_sas0_initial(y[0], sa_list[0], s_0)[0]) + policy_sum


def log_p_theta_obs(y, sa_list, s_0, theta):
    """
    :param y: the sequence of observations given states and sensing actions
    :param sa_list: the sequence of sensing actions
    :param s_0: the initial state
    :param theta: the sensing policy parameter
    :return: the log probability log P(y, sa_list; theta)
    """
    m = fsc.memory_space.index('l')
    policy_sum = np.log2(pi_theta(m, sa_list[0], theta))
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_sum += np.log2(pi_theta(m, sa_list[i + 1], theta))
    return np.log2(p_obs_g_sas0(y, sa_list, s_0)[1]) - np.log2(
        p_obs_g_sas0_initial(y[0], sa_list[0], s_0)[1]) + policy_sum


def nabla_log_p_theta_obs_g_s0(y, sa_list, theta):
    m = fsc.memory_space.index('l')
    log_grad_sum = log_policy_gradient(m, sa_list[0], theta)
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grad_sum += log_policy_gradient(m, sa_list[i + 1], theta)
    return log_grad_sum


def nabla_p_theta_obs_g_s0(y, sa_list, s_0, theta):
    m = fsc.memory_space.index('l')
    log_grad_sum = log_policy_gradient(m, sa_list[0], theta)
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grad_sum += log_policy_gradient(m, sa_list[i + 1], theta)
    return p_theta_obs_g_s0(y, sa_list, s_0, theta) * log_grad_sum


def p_theta_obs(y, sa_list, theta):
    p_obs = 0
    for state_0 in env.initial_states:
        s_0 = env.states.index(state_0)
        p_obs += p_theta_obs_g_s0(y, sa_list, s_0, theta) * env.initial_state_dis[s_0, 0]
    return p_obs


def nabla_p_theta_obs(y, sa_list, theta):
    nabla_p_obs = 0
    for state_0 in env.initial_states:
        s_0 = env.states.index(state_0)
        nabla_p_obs += env.initial_state_dis[s_0, 0] * nabla_p_theta_obs_g_s0(y, sa_list, s_0, theta)
    return nabla_p_obs


# def nabla_log_p_theta_obs(y, sa_list, theta):
#     return nabla_p_theta_obs(y, sa_list, theta) / p_theta_obs(y, sa_list, theta)


def nabla_log_p_theta_obs(y, sa_list, theta):
    m = fsc.memory_space.index('l')
    log_grad_sum = log_policy_gradient(m, sa_list[0], theta)
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grad_sum += log_policy_gradient(m, sa_list[i + 1], theta)
    return log_grad_sum


def log_p_theta_s0_g_y(y, sa_list, s_0, theta):
    log_p_theta_y_g_s0 = log_p_theta_obs_g_s0(y, sa_list, s_0, theta)
    log_p_theta_y = log_p_theta_obs(y, sa_list, s_0, theta)
    return log_p_theta_y_g_s0 + np.log2(env.initial_state_dis[s_0, 0]) - log_p_theta_y


# def p_theta_s0_g_y(y, sa_list, s_0, theta):
#     return p_theta_obs_g_s0(y, sa_list, s_0, theta) * env.initial_state_dis[s_0, 0] / p_theta_obs(y, sa_list, theta)


def p_theta_s0_g_y(y, sa_list, s_0, theta):
    return 2 ** log_p_theta_s0_g_y(y, sa_list, s_0, theta)


def nabla_p_theta_s0_g_y(y, sa_list, s_0, theta):
    const = env.initial_state_dis[s_0, 0] / (p_theta_obs(y, sa_list, theta) ** 2)
    grad_diff = (nabla_p_theta_obs_g_s0(y, sa_list, s_0, theta) * p_theta_obs(y, sa_list, theta)
                 - p_theta_obs_g_s0(y, sa_list, s_0, theta) * nabla_p_theta_obs(y, sa_list, theta))
    return const * grad_diff


def entropy_a_grad(y_data, sa_data, theta):  # only for debugging
    M = len(y_data)
    H = 0
    nabla_H = np.zeros([fsc.memory_size, env.sensing_actions_size])
    for k in range(M):
        y_k = y_data[k]
        sa_list_k = sa_data[k]
        # Get the values when z_T = 1
        # p_theta_yk = p_theta_obs(y_k, sa_list_k, theta)
        grad_log_P_theta_yk = nabla_log_p_theta_obs(y_k, sa_list_k, theta)

        for state_0 in env.initial_states:
            s_0 = env.states.index(state_0)
            p_theta_s0_yk = p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
            # log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
            temp_H = p_theta_s0_yk * np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
            H += temp_H
            # grad_p_theta_s0_yk = nabla_p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
            nabla_H += temp_H * grad_log_P_theta_yk
    H = - H / M
    nabla_H = - nabla_H / M
    return H, nabla_H


def entropy_a_grad_per_iter(y_k, sa_list_k, theta):
    # Get the values when z_T = 1
    # p_theta_yk = p_theta_obs(y_k, sa_list_k, theta)
    grad_log_P_theta_yk = nabla_log_p_theta_obs(y_k, sa_list_k, theta)
    H_per_iter = 0
    nabla_H_per_iter = np.zeros([fsc.memory_size, env.sensing_actions_size])

    for state_0 in env.initial_states:
        s_0 = env.states.index(state_0)
        p_theta_s0_yk = p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
        log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
        temp_H = p_theta_s0_yk * log2_p_theta_s0_yk if p_theta_s0_yk > 0 else 0
        H_per_iter += temp_H
        # grad_p_theta_s0_yk = nabla_p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
        nabla_H_per_iter += temp_H * grad_log_P_theta_yk
    return H_per_iter, nabla_H_per_iter


def entropy_a_grad_multi(y_data, sa_data, theta):
    M = len(y_data)
    H = 0
    nabla_H = np.zeros([fsc.memory_size, env.sensing_actions_size])
    with ProcessPoolExecutor(max_workers=24) as exe:
        H_a_gradH_list = exe.map(entropy_a_grad_per_iter, y_data, sa_data, repeat(theta))
    for H_tuple in H_a_gradH_list:
        H += H_tuple[0]
        nabla_H += H_tuple[1]
    H = - H / M
    nabla_H = - nabla_H / M
    return H, nabla_H


def action_sampler(s):
    prob_list = [env.policy[s, a] for a in range(env.action_size)]
    return choices(env.actions, prob_list, k=1)[0]


def sensing_action_sampler(theta, m):
    prob_list = [pi_theta(m, sa, theta) for sa in range(env.sensing_actions_size)]
    return choices(env.sensing_actions, prob_list, k=1)[0]


def sample_data(M, T, theta):
    s_data = np.zeros([M, T], dtype=np.int32)
    a_data = np.zeros([M, T], dtype=np.int32)
    sa_data = []
    y_data = []
    for m in range(M):
        y = []
        sa_list = []
        # start from initial state
        state = choice(env.initial_states)
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


def main():
    # Define hyperparameters
    ex_num = 3
    iter_num = 1000  # iteration number of gradient ascent
    M = 2000  # number of sampled trajectories
    T = 10  # length of a trajectory
    eta = 0.2  # step size for theta
    # kappa = 0.2  # step size for lambda
    # F = env.goals  # Define the goal region
    # alpha = 0.3  # value constraint

    # Initialize the parameters
    theta = np.random.random([fsc.memory_size, env.sensing_actions_size])
    # opt_values = value_iterations(1e-3, F)
    # theta = extract_opt_theta(opt_values, F)  # optimal theta initialization.
    # with open('backward_grid_world_1_data/Values/theta_3', 'rb') as f:
    #     theta = np.load(f, allow_pickle=True)

    # lam = np.random.uniform(1, 10)
    # Create empty lists
    entropy_list = []
    # value_list = []
    # Sample trajectories (observations)
    for i in range(iter_num):
        start = time.time()
        ##############################################
        s_data, a_data, y_data, sa_data = sample_data(M, T, theta)
        # Gradient ascent process
        # print(y_data)
        # SGD gradient
        approx_entropy, grad_H = entropy_a_grad_multi(y_data, sa_data, theta)
        # print(grad_H)
        # print("The gradient of entropy is", grad_H)
        print("The conditional entropy is", approx_entropy)
        entropy_list.append(approx_entropy)
        # SGD updates
        theta = theta - eta * grad_H
        # lam = lam - kappa * (approx_value - alpha)
        ###############################################
        end = time.time()
        print("One iteration done. It takes", end - start, "s")

    with open(f'./grid_world_1_data/Values/theta_{ex_num}.npy', 'wb') as f:
        np.save(f, theta)

    with open(f'./grid_world_1_data/Values/entropy_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(entropy_list, pkl_wb_obj)

    iteration_list = range(iter_num)
    plt.plot(iteration_list, entropy_list, label='entropy')
    plt.xlabel("The iteration number")
    plt.ylabel("entropy")
    plt.legend()
    plt.savefig(f'./grid_world_1_data/Graphs/Ex_{ex_num}_iter1k_M2000_T10.png')
    plt.show()


if __name__ == "__main__":
    main()
