import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from math import isinf

from random import choices, choice
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from grid_world_2_obstacle import Environment
from finite_state_controller import FSC
from policy_network import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Environment()
fsc = FSC()

PRIOR = env.get_prior_distribution([0.1, 0.4, 0.5])
# PRIOR = torch.from_numpy(PRIOR).type(dtype=torch.float32)
# PRIOR = PRIOR.to(device)


def observable_operator(o_t, a_t):
    oo = np.zeros([env.state_size, env.state_size])
    for i in range(env.state_size):
        for j in range(env.state_size):
            # The definition of observable operators
            oo[i, j] = env.transition_wc[j, i] * env.emission_function(j, a_t, o_t)
    return oo


def p_obs_g_sas0(y, sa_list, s_0):
    # Get the real sensing actions
    senAct_list = [env.sensing_actions[sa] for sa in sa_list]
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
    # print(y)
    # print(senAct_list)
    # print(probs)
    probs_1 = probs @ mu_0
    probs_2 = probs @ PRIOR
    return probs_1[0][0], probs_2[0][0]


def p_obs_g_sas0_initial(o_0, sa_0, s_0):
    # Get real sensing action
    senAct_0 = env.sensing_actions[sa_0]
    mu_0 = np.zeros([env.state_size, 1])
    mu_0[s_0, 0] = 1
    # Obtain observable operators
    oo = observable_operator(o_0, senAct_0)
    # Creat a vector with all elements equals to 1
    one_vec = np.ones([1, env.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    probs_1 = probs @ mu_0
    probs_2 = probs @ PRIOR
    return probs_1[0][0], probs_2[0][0]


def p_theta_obs_g_s0(policy_net, y, sa_list, s_0):
    """
    :param y: the sequence of observations given states and sensing actions
    :param sa_list: the sequence of sensing actions
    :param s_0: the initial state
    :param theta: the sensing policy parameter
    :return: the probability P(y, sa_list| s_0 ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_prod = get_action_probability(policy_net, m, sa_list[0])
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_prod *= get_action_probability(policy_net, m, sa_list[i + 1])
    return p_obs_g_sas0(y, sa_list, s_0)[0] / p_obs_g_sas0_initial(y[0], sa_list[0], s_0)[0] * policy_prod


def log_p_theta_obs_g_s0(policy_net, y, sa_list, s_0):
    """
    :param y: the sequence of observations given states and sensing actions
    :param sa_list: the sequence of sensing actions
    :param s_0: the initial state
    :param theta: the sensing policy parameter
    :return: the log probability log P(y, sa_list| s_0 ; theta)
    """
    m = fsc.memory_space.index('l')
    policy_sum = np.log2(get_action_probability(policy_net, m, sa_list[0]))
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_sum += np.log2(get_action_probability(policy_net, m, sa_list[i + 1]))
    log_p_y_g_sas0 = np.log2(p_obs_g_sas0(y, sa_list, s_0)[0]) if p_obs_g_sas0(y, sa_list, s_0)[0] > 0 else float(
        '-inf')
    # print(log_p_y_g_sas0)
    return log_p_y_g_sas0 - np.log2(p_obs_g_sas0_initial(y[0], sa_list[0], s_0)[0]) + policy_sum


def log_p_theta_obs(policy_net, y, sa_list, s_0):
    """
    :param y: the sequence of observations given states and sensing actions
    :param sa_list: the sequence of sensing actions
    :param s_0: the initial state
    :param theta: the sensing policy parameter
    :return: the log probability log P(y, sa_list; theta)
    """
    m = fsc.memory_space.index('l')
    policy_sum = np.log2(get_action_probability(policy_net, m, sa_list[0]))
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        policy_sum += np.log2(get_action_probability(policy_net, m, sa_list[i + 1]))
    log_p_y_g_sa = np.log2(p_obs_g_sas0(y, sa_list, s_0)[1]) if p_obs_g_sas0(y, sa_list, s_0)[1] > 0 else float('-inf')
    # print(log_p_y_g_sa)
    return log_p_y_g_sa - np.log2(
        p_obs_g_sas0_initial(y[0], sa_list[0], s_0)[1]) + policy_sum


def nabla_log_p_theta_obs_g_s0(policy_net, y, sa_list):
    m = fsc.memory_space.index('l')
    log_grad_sum = create_gradient_shaped_tensors(policy_net)
    log_grads = compute_log_policy_gradient(policy_net,  m, sa_list[0])
    for j in range(len(log_grad_sum)):
        log_grad_sum[j] = log_grads[j]
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grads = compute_log_policy_gradient(policy_net, m, sa_list[i + 1])
        for j in range(len(log_grad_sum)):
            log_grad_sum[j] += log_grads[j]
    return log_grad_sum


def nabla_p_theta_obs_g_s0(policy_net, y, sa_list, s_0):
    m = fsc.memory_space.index('l')
    log_grad_sum = create_gradient_shaped_tensors(policy_net)
    log_grads = compute_log_policy_gradient(policy_net, m, sa_list[0])
    for j in range(len(log_grad_sum)):
        log_grad_sum[j] = log_grads[j]
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grads = compute_log_policy_gradient(policy_net, m, sa_list[i + 1])
        for j in range(len(log_grad_sum)):
            log_grad_sum[j] += log_grads[j]
    for j in range(len(log_grad_sum)):
        log_grad_sum[j] = p_theta_obs_g_s0(policy_net, y, sa_list, s_0) * log_grad_sum[j]
    return log_grad_sum


def p_theta_obs(policy_net, y, sa_list):
    p_obs = 0
    for state_0 in env.initial_states:
        s_0 = env.states.index(state_0)
        p_obs += p_theta_obs_g_s0(policy_net, y, sa_list, s_0) * PRIOR[s_0, 0]
    return p_obs


def nabla_p_theta_obs(policy_net, y, sa_list):
    nabla_p_obs = create_gradient_shaped_tensors(policy_net)
    for state_0 in env.initial_states:
        s_0 = env.states.index(state_0)
        grads_p_theta_obs_g_s0 = nabla_p_theta_obs_g_s0(policy_net, y, sa_list, s_0)
        for j in range(len(nabla_p_obs)):
            nabla_p_obs[j] += PRIOR[s_0, 0] * grads_p_theta_obs_g_s0[j]
    return nabla_p_obs


# def nabla_log_p_theta_obs(y, sa_list, theta):
#     return nabla_p_theta_obs(y, sa_list, theta) / p_theta_obs(y, sa_list, theta)


def nabla_log_p_theta_obs(policy_net, y, sa_list):
    m = fsc.memory_space.index('l')
    log_grad_sum = create_gradient_shaped_tensors(policy_net)
    log_grads = compute_log_policy_gradient(policy_net, m, sa_list[0])
    for j in range(len(log_grad_sum)):
        log_grad_sum[j] = log_grads[j]
    for i in range(len(y) - 1):
        o = fsc.observations.index(y[i])
        m = fsc.transition[m][o]
        log_grads = compute_log_policy_gradient(policy_net, m, sa_list[i + 1])
        for j in range(len(log_grad_sum)):
            log_grad_sum[j] += log_grads[j]
    return log_grad_sum


def p_theta_s0_g_y_stable(policy_net, y, sa_list, scale_factor=1e10):
    prob_list = np.zeros(len(env.initial_states))
    for i in range(len(env.initial_states)):
        state_0 = env.initial_states[i]
        s_0 = env.states.index(state_0)
        # Highly depends on the prior distribution
        # print(state_0)
        p_theta_y_s0 = p_theta_obs_g_s0(policy_net, y, sa_list, s_0)
        # print(p_theta_y_s0)
        prob_list[i] = p_theta_y_s0 * PRIOR[s_0, 0]
    scaled_prob_list = [x * scale_factor for x in prob_list]
    total = sum(scaled_prob_list)
    if total == 0:
        normalized_list = [1 / len(prob_list) for i in range(len(prob_list))]
    else:
        normalized_list = [x / total for x in scaled_prob_list]
    return normalized_list


def log_p_theta_s0_g_y(policy_net, y, sa_list, s_0):
    log_p_theta_y_g_s0 = log_p_theta_obs_g_s0(policy_net, y, sa_list, s_0)
    log_p_theta_y = log_p_theta_obs(policy_net, y, sa_list, s_0)
    if isinf(log_p_theta_y_g_s0) or isinf(log_p_theta_y):
        return float('-inf')
    else:
        return log_p_theta_y_g_s0 + np.log2(PRIOR[s_0, 0]) - log_p_theta_y


# def p_theta_s0_g_y(y, sa_list, s_0, theta):
#     return p_theta_obs_g_s0(y, sa_list, s_0, theta) * PRIOR[s_0, 0] / p_theta_obs(y, sa_list, theta)


def p_theta_s0_g_y(policy_net, y, sa_list, s_0): # summation not equal to 1!
    return 2 ** log_p_theta_s0_g_y(policy_net, y, sa_list, s_0)


def nabla_p_theta_s0_g_y(policy_net, y, sa_list, s_0):
    const = PRIOR[s_0, 0] / (p_theta_obs(policy_net, y, sa_list) ** 2)
    grad_diff = create_gradient_shaped_tensors(policy_net)
    grads_p_theta_obs_g_s0 = nabla_p_theta_obs_g_s0(policy_net, y, sa_list, s_0)
    grads_p_theta_obs = nabla_p_theta_obs(policy_net, y, sa_list)
    for j in range(len(grad_diff)):
        grad_diff[j] = const * (grads_p_theta_obs_g_s0[j] * p_theta_obs(policy_net, y, sa_list)
                     - p_theta_obs_g_s0(policy_net, y, sa_list, s_0) * grads_p_theta_obs[j])
    return grad_diff


def entropy_a_grad(policy_net, y_data, sa_data):  # only for debugging
    M = len(y_data)
    H = 0
    nabla_H = create_gradient_shaped_tensors(policy_net)
    for k in range(M):
        y_k = y_data[k]
        sa_list_k = sa_data[k]
        # Get the values when z_T = 1
        # p_theta_yk = p_theta_obs(y_k, sa_list_k, theta)
        grad_log_P_theta_yk = nabla_log_p_theta_obs(policy_net, y_k, sa_list_k)
        p_theta_s0_yk_list = p_theta_s0_g_y_stable(policy_net, y_k, sa_list_k)
        # print('The posterior distribution is', p_theta_s0_yk_list)
        # print('#'*100)

        for i in range(len(env.initial_states)):
            p_theta_s0_yk = p_theta_s0_yk_list[i]
            # log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
            temp_H = p_theta_s0_yk * np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
            H += temp_H
            # grad_p_theta_s0_yk = nabla_p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
            for j in range(len(nabla_H)):
                nabla_H[j] += temp_H * grad_log_P_theta_yk[j]
    H = - H / M
    for j in range(len(nabla_H)):
        nabla_H[j] = - nabla_H[j] / M
    return H, nabla_H


# def entropy_a_grad_per_iter(y_k, sa_list_k, theta):
#     # Get the values when z_T = 1
#     # p_theta_yk = p_theta_obs(y_k, sa_list_k, theta)
#     grad_log_P_theta_yk = nabla_log_p_theta_obs(y_k, sa_list_k, theta)
#     p_theta_s0_yk_list = p_theta_s0_g_y_stable(y_k, sa_list_k, theta)
#     H_per_iter = 0
#     nabla_H_per_iter = np.zeros([fsc.memory_size, env.sensing_actions_size])
#
#     for i in range(len(env.initial_states)):
#         p_theta_s0_yk = p_theta_s0_yk_list[i]
#         # print(p_theta_s0_yk)
#         log2_p_theta_s0_yk = np.log2(p_theta_s0_yk) if p_theta_s0_yk > 0 else 0
#         temp_H = p_theta_s0_yk * log2_p_theta_s0_yk if p_theta_s0_yk > 0 else 0
#         H_per_iter += temp_H
#         # grad_p_theta_s0_yk = nabla_p_theta_s0_g_y(y_k, sa_list_k, s_0, theta)
#         nabla_H_per_iter += temp_H * grad_log_P_theta_yk
#     # print(H_per_iter)
#     return H_per_iter, nabla_H_per_iter
#
#
# def entropy_a_grad_multi(y_data, sa_data, theta):
#     M = len(y_data)
#     H = 0
#     nabla_H = np.zeros([fsc.memory_size, env.sensing_actions_size])
#     with ProcessPoolExecutor(max_workers=24) as exe:
#         H_a_gradH_list = exe.map(entropy_a_grad_per_iter, y_data, sa_data, repeat(theta))
#     for H_tuple in H_a_gradH_list:
#         H += H_tuple[0]
#         nabla_H += H_tuple[1]
#     H = - H / M
#     nabla_H = - nabla_H / M
#     return H, nabla_H


def action_sampler(s):
    prob_list = [env.policy[s, a] for a in range(env.action_size)]
    return choices(env.actions, prob_list, k=1)[0]


def sensing_action_sampler(policy_net, m):
    prob_list = [get_action_probability(policy_net, m, sa) for sa in range(env.sensing_actions_size)]
    return choices(env.sensing_actions, prob_list, k=1)[0]


def sample_data(policy_net, M, T):
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
        sAct = sensing_action_sampler(policy_net, me)
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
            sAct = sensing_action_sampler(policy_net, me)
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
    ex_num = 1
    iter_num = 2000  # iteration number of gradient ascent
    M = 1000  # number of sampled trajectories
    T = 10  # length of a trajectory
    eta = 0.01  # step size for theta
    # kappa = 0.2  # step size for lambda
    # F = env.goals  # Define the goal region
    # alpha = 0.3  # value constraint
    state_dim = 1
    hidden_dim = 64

    policy_net = PolicyNetwork(state_dim, env.sensing_actions_size, hidden_dim)
    test_grads = create_gradient_shaped_tensors(policy_net)

    # Initialize the parameters
    # theta = np.random.random([fsc.memory_size, env.sensing_actions_size])
    # opt_values = value_iterations(1e-3, F)
    # theta = extract_opt_theta(opt_values, F)  # optimal theta initialization.
    # with open('backward_grid_world_1_data/Values/theta_3', 'rb') as f:
    #     theta = np.load(f, allow_pickle=True)

    # lam = np.random.uniform(1, 10)
    # Create empty lists
    entropy_list = []
    # theta_list = [theta]
    # Sample trajectories (observations)
    for i in range(iter_num):
        start = time.time()
        ##############################################
        s_data, a_data, y_data, sa_data = sample_data(policy_net, M, T)
        # Gradient ascent process
        # print(y_data)
        # SGD gradient
        approx_entropy, grad_H = entropy_a_grad(policy_net, y_data, sa_data)
        # grad_H = torch.from_numpy(grad_H).type(dtype=torch.float32)
        # print("The gradient of entropy is", grad_H)
        print("The conditional entropy is", approx_entropy)
        entropy_list.append(approx_entropy)
        # SGD updates
        counter = 0
        for param in policy_net.parameters():
            if param.grad is not None:  # Ensure the gradient exists
                with torch.no_grad():
                    param += -eta * grad_H[counter]  # Tensor of zeros
                counter += 1
        # theta = theta - eta * grad_H
        # theta_list.append(theta)
        ###############################################
        end = time.time()
        print(f"iteration_{i+1} done. It takes", end - start, "s")
        print("#" * 100)

    # with open(f'./grid_world_2_data/Values/Correct_thetaList_{ex_num}', "wb") as pkl_wb_obj:
    #     pickle.dump(theta_list, pkl_wb_obj)

    with open(f'./deep_data/Values/Correct_entropy_{ex_num}', "wb") as pkl_wb_obj:
        pickle.dump(entropy_list, pkl_wb_obj)

    iteration_list = range(iter_num)
    plt.plot(iteration_list, entropy_list, label='entropy')
    plt.xlabel("The iteration number")
    plt.ylabel("entropy")
    plt.legend()
    plt.savefig(f'./deep_data/Graphs/CorrectEx_{ex_num}_dynaNoi01_obsNoi01.png')
    plt.show()


if __name__ == "__main__":
    main()
