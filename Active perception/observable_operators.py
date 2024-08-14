import itertools
import numpy as np
from grid_world_1 import Environment

env = Environment()


def permutations_with_repetition(elements, length):
    # Generate permutations with repetition
    results = itertools.product(elements, repeat=length)

    # Convert each tuple to a string and collect into a list
    string_results = [''.join(result) for result in results]

    return string_results


def get_memory_space(K):
    memory_space = permutations_with_repetition(env.observations, K)
    memory_size = env.observations_size ** K
    return memory_space, memory_size


def pi_theta(m, a, theta):
    """
    :param m: the index of a finite sequence of observation, corresponding to K-step memory
    :param a: the sensing action to be given
    :param theta: the policy parameter, the size state_size^3 * sensing_action_size
    :return: the Gibbs policy given the finite memory
    """
    e_x = np.exp(theta[m, :] - np.max(theta[m, :]))
    return (e_x / e_x.sum(axis=0))[a]


def log_policy_gradient(K, m, sa, theta):
    # A memory space for K-step memory policy
    memory_space, memory_size = get_memory_space(K)
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


def p_obs_g_sas0(y, sa_list, s_0):
    # Give value to the initial state
    mu_0 = np.zeros([env.state_size, 1])
    mu_0[s_0, 0] = 1
    # Obtain observable operators
    oo = observable_operator(y[0], sa_list[0])
    # Creat a vector with all elements equals to 1
    one_vec = np.zeros([1, env.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo
    # Calculate the probability
    for t in range(1, len(y)):
        oo = observable_operator(y[t], sa_list[t])
        probs = probs @ oo
    probs = probs @ mu_0
    return 0


def p_obs_g_sas0_initial(o_0, sa_0, s_0):
    mu_0 = np.zeros([env.state_size, 1])
    mu_0[s_0, 0] = 1
    # Obtain observable operators
    oo = observable_operator(o_0, sa_0)
    # Creat a vector with all elements equals to 1
    one_vec = np.zeros([1, env.state_size])
    # Initialize the probability of observation given sensing actions and initial states
    probs = one_vec @ oo @ mu_0
    return probs
