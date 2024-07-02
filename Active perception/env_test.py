import numpy as np
from grid_world_1 import Environment
from random import choices

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
    y_data = []
    for m in range(M):
        y = []
        # start from initial state
        state = env.initial_state
        # Get the observation of initial state
        y.append(env.observation_function(state)[0])
        # Sample the action from initial state
        act = action_sampler(theta, state)
        for t in range(T):
            s = env.states.index(state)
            s_data[m, t] = s
            a = env.actions.index(act)
            a_data[m, t] = a
            # sample the next state
            state = env.next_state_sampler(state, act)
            # Add the observation
            y.append(env.observation_function(state)[0])
            # sample action
            act = action_sampler(theta, state)
        y_data.append(y)
    return s_data, a_data, y_data


theta = np.ones([env.state_size, env.action_size])
s_data, a_data, y_data = sample_data(1, 20, theta)
for i in range(len(s_data[0])):
    print("The state is", env.states[s_data[0][i]])
    print("The action is", env.actions[a_data[0][i]])
    print("The observation is", y_data[0][i])
