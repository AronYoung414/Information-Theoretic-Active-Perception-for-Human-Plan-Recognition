from random import choices
import numpy as np

WIDTH = 4
LENGTH = 4


class Environment:

    def __init__(self):
        # parameter which controls environment noise
        self.stoPar = 0.1
        # parameter which controls observation noise
        # self.obs_noise = 1 / 3
        # Define states
        self.states = [(i, j) for i in range(WIDTH) for j in range(LENGTH)]
        # self.state_indices = list(range(len(self.states)))
        self.state_size = len(self.states)
        # Define initial state
        self.initial_state = (2, 0)
        self.initial_state_idx = self.states.index(self.initial_state)
        # Define actions
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.action_size = len(self.actions)
        # self.action_indices = list(range(len(self.actions)))
        # Goals
        self.goals_1 = [(0, 3)]  # The goal of type 1 agent
        self.goals_2 = [(3, 3)]  # The goal of type 2 agent
        # transition probability dictionary
        self.transition = self.get_transition()
        # Define observations
        self.observations = ['b', 'r', 'n']
        # Define sensors
        self.sensors = [[(0, 2), (1, 2)],
                        [(2, 2), (3, 2)]]
        # The optimal policies
        self.value_1 = self.value_iterations(0.01, self.goals_1)
        self.policy_1 = self.optimal_policy(self.value_1, self.goals_1)
        self.value_2 = self.value_iterations(0.01, self.goals_2)
        self.policy_2 = self.optimal_policy(self.value_2, self.goals_2)
        # Obtain transition probabilities
        self.transition_1 = self.get_transition_prob(self.policy_1)
        self.transition_2 = self.get_transition_prob(self.policy_2)

    def complementary_actions(self, act):
        # Use to find out stochastic transitions, if it stays, no stochasticity, if other actions, return possible stochasticity directions.
        if act == (0, 0):
            return []
        elif act[0] == 0:
            return [(1, 0), (-1, 0)]
        else:
            return [(0, 1), (0, -1)]

    def check_inside(self, st):
        # If the state is valid or not
        if st in self.states:
            return True
        return False

    def get_transition(self):
        # Constructing transition function trans[state][action][next_state] = probability
        stoPar = self.stoPar
        trans = {}
        for st in self.states:
            trans[st] = {}
            for act in self.actions:
                if act == (0, 0):
                    trans[st][act] = {}
                    trans[st][act][st] = 1
                else:
                    trans[st][act] = {}
                    trans[st][act][st] = 0
                    tempst = tuple(np.array(st) + np.array(act))
                    if self.check_inside(tempst):
                        trans[st][act][tempst] = 1 - 2 * stoPar
                    else:
                        trans[st][act][st] += 1 - 2 * stoPar
                    for act_ in self.complementary_actions(act):
                        tempst_ = tuple(np.array(st) + np.array(act_))
                        if self.check_inside(tempst_):
                            trans[st][act][tempst_] = stoPar
                        else:
                            trans[st][act][st] += stoPar
        # self.check_trans(trans)
        return trans

    def get_transition_prob(self, policy):
        trans_prob = np.zeros([self.state_size, self.state_size])
        for st in self.states:
            for act in self.actions:
                s = self.states.index(st)
                a = self.actions.index(act)
                state_p_dict = self.transition[st][act]
                for st_p in state_p_dict.keys():
                    s_p = self.states.index(st_p)
                    s_prime_prob = state_p_dict[st_p]
                    trans_prob[s, s_p] += s_prime_prob * policy[s, a]
        return trans_prob

    def check_trans(self, trans):
        # Check if the transitions are constructed correctly
        for st in trans.keys():
            for act in trans[st].keys():
                if abs(sum(trans[st][act].values()) - 1) > 0.01:
                    print("st is:", st, "act is:", act, "sum is:", sum(self.transition[st][act].values()))
                    return False
        print("Transition is correct")
        return True

    def next_state_sampler(self, state, act):
        next_supp = list(self.transition[state][act].keys())
        next_prob = [self.transition[state][act][next_s] for next_s in next_supp]
        next_state = choices(next_supp, next_prob)[0]
        return next_state

    def observation_function(self, state):
        if state in self.sensors[0]:
            return ['b']
        elif state in self.sensors[1]:
            return ['r']
        else:
            return ['n']

    # def observation_function_sampler(self, state):
    #     observation_set = self.observation_function(state)
    #     if len(observation_set) > 1:
    #         return choices(observation_set, [1 - self.obs_noise, self.obs_noise], k=1)[0]
    #     else:
    #         return self.observations[-1]

    def emission_function(self, state, o):
        observation_set = self.observation_function(state)
        if o in observation_set:
            return 1
        else:
            return 0

    def get_reward(self, state, F):
        if state in F:
            return 1
        else:
            return 0

    def value_iterations(self, threshold, goal, gamma=0.8):
        """
        :param goal: indicate the type of agent
        :param threshold: threshold for Bellman error
        :param gamma: discount rate
        :return: value function
        """
        values = np.zeros(self.state_size)
        values_old = np.copy(values)
        Delta = threshold + 0.1
        while Delta > threshold:
            for state in self.states:
                v_n = 0
                for act in self.actions:
                    state_p_dict = self.transition[state][act]
                    temp_v = 0
                    for state_p in state_p_dict.keys():
                        s_p_prob = state_p_dict[state_p]
                        s_p = self.states.index(state_p)
                        temp_v += s_p_prob * (self.get_reward(state, goal) + gamma * values[s_p])
                    if temp_v > v_n:
                        v_n = temp_v
                s = self.states.index(state)
                values[s] = v_n
            Delta = np.max(values - values_old)
            values_old = np.copy(values)
        return values

    def optimal_policy(self, opt_values, goal, tau=0.01, gamma=0.8):
        pi_star = np.zeros([self.state_size, self.action_size])
        for state in self.states:
            for act in self.actions:
                state_p_dict = self.transition[state][act]
                next_v = 0
                for state_p in state_p_dict.keys():
                    s_prime_prob = state_p_dict[state_p]
                    s_p = self.states.index(state_p)
                    next_v += s_prime_prob * (self.get_reward(state, goal) + gamma * opt_values[s_p])
                s = self.states.index(state)
                a = self.actions.index(act)
                pi_star[s, a] = np.exp(next_v / tau) / np.exp(opt_values[s] / tau)
        return pi_star
