import numpy as np
from grid_world_2_obstacle import Environment
from random import choices
from random import choice

env = Environment()


def pi_theta(state, act, theta):
    # Gibbs policy
    s = env.states.index(state)
    a = env.actions.index(act)
    e_x = np.exp(theta[s, :] - np.max(theta[s, :]))
    return (e_x / e_x.sum(axis=0))[a]


def action_sampler(theta, state):
    prob_list = [pi_theta(state, act, theta) for act in env.actions]
    return choices(env.actions, prob_list, k=1)[0]


def sample_data(M, T, theta):
    s_data = np.zeros([M, T], dtype=np.int32)
    a_data = np.zeros([M, T], dtype=np.int32)
    sa_data = []
    y_data = []
    ep_data = np.zeros([M, T])
    for m in range(M):
        y = []
        sa_list = []
        # start from initial state
        state = env.initial_states[0]
        # Sample sensing action
        sAct = choice(env.sensing_actions)
        sa_list.append(sAct)
        # Get the observation of initial state
        o = env.observation_function_sampler(state, sAct)
        y.append(o)
        # Sample the action from initial state
        act = action_sampler(theta, state)

        for t in range(T):
            s = env.states.index(state)
            s_data[m, t] = s
            a = env.actions.index(act)
            a_data[m, t] = a
            # The emission probability
            ep_data[m, t] = env.emission_function(s, sAct, o)
            # sample the next state
            state = env.next_state_sampler(state, act)
            # Sample sensing action
            sAct = choice(env.sensing_actions)
            sa_list.append(sAct)
            # Add the observation
            o = env.observation_function_sampler(state, sAct)
            y.append(o)
            # sample action
            act = action_sampler(theta, state)

        y_data.append(y)
        sa_data.append(sa_list)
    return s_data, a_data, y_data, sa_data, ep_data


opt_value = env.get_optimal_policy(env.value, env.goals)
theta = env.extract_opt_theta(env.value, env.goals)
s_data, a_data, y_data, sa_data, ep_data = sample_data(1, 30, theta)
for i in range(len(s_data[0])):
    print("The state is", env.states[s_data[0][i]])
    print("The action is", env.actions[a_data[0][i]])
    print("The observation is", y_data[0][i])
    print("The sensing action is", sa_data[0][i])
    print("The emission probability is", ep_data[0][i])
    print("#" * 30)
