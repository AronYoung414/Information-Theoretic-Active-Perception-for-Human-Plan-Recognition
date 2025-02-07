from random import choices
import numpy as np

WIDTH = 6
LENGTH = 6


class Environment:

    def __init__(self):
        # parameter which controls environment noise
        self.stoPar = 0.1
        # parameter which controls observation noise
        self.obs_noise = 0.1
        # Define obstacles
        self.obstacles = [(2, 1), (5, 1), (0, 2), (3, 3)]
        # Define states
        self.whole_states = [(i, j) for i in range(WIDTH) for j in range(LENGTH)]
        self.states = list(set(self.whole_states) - set(self.obstacles))
        # self.state_indices = list(range(len(self.states)))
        self.state_size = len(self.states)
        # Define initial state
        self.initial_states = [(3, 0), (0, 3), (5, 2)]
        self.initial_state_dis = self.get_initial_distribution()
        # Define actions
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.action_size = len(self.actions)
        # self.action_indices = list(range(len(self.actions)))
        # Goals
        self.goals = [(0, 5), (5, 5)]  # The goal of agent
        # transition probability dictionary
        self.transition = self.get_transition()
        # Define observations
        self.observations = ['1', '2', '3', '4', '5', 'n']
        self.observations_size = len(self.observations)
        # Define sensors
        self.sensors = [[(1, 3), (1, 4), (2, 3), (2, 4)],
                        [(4, 3), (4, 4), (5, 3), (5, 4)],
                        [(3, 1), (3, 2), (4, 1), (4, 2)],
                        [(1, 5), (2, 5), (3, 5), (4, 5)],
                        [(0, 1), (1, 1), (1, 2), (2, 2)]]
        # Define sensing actions
        self.sensing_actions = ['1', '2', '3', '4', '5']
        self.sensing_actions_size = len(self.sensing_actions)
        # The optimal policies
        self.value = self.value_iterations(0.01, self.goals)
        self.policy = self.get_optimal_policy(self.value, self.goals)
        # Obtain transition probabilities
        self.transition_wc = self.get_transition_wc(self.policy)

    def get_prior_distribution(self, prior):
        initial_dis = np.zeros([self.state_size, 1])
        for i in range(len(self.initial_states)):
            state_0 = self.initial_states[i]
            s_0 = self.states.index(state_0)
            initial_dis[s_0] = prior[i]
        return initial_dis

    def get_initial_distribution(self):
        initial_dis = np.zeros([self.state_size, 1])
        for s in range(self.state_size):
            if self.states[s] in self.initial_states:
                initial_dis[s, 0] = 1 / len(self.initial_states)  # uniform distribution
        return initial_dis

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

    def get_transition_wc(self, policy):
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

    def observation_function(self, state, sAct):
        if state in self.sensors[0] and self.observations[0] == sAct:
            return [self.observations[0], self.observations[5]]
        elif state in self.sensors[1] and self.observations[1] == sAct:
            return [self.observations[1], self.observations[5]]
        elif state in self.sensors[2] and self.observations[2] == sAct:
            return [self.observations[2], self.observations[5]]
        elif state in self.sensors[3] and self.observations[3] == sAct:
            return [self.observations[3], self.observations[5]]
        elif state in self.sensors[4] and self.observations[4] == sAct:
            return [self.observations[4], self.observations[5]]
        else:
            return [self.observations[5]]

    # def emission_function(self, s, sAct, o):
    #     state = self.states[s]
    #     observation_set = self.observation_function(state, sAct)
    #     if o in observation_set:
    #         return 1
    #     else:
    #         return 0

    def emission_function(self, s, sAct, o):
        state = self.states[s]
        observation_set = self.observation_function(state, sAct)
        if o in observation_set:
            if o == self.observations[5] and len(observation_set) == 1:
                return 1
            elif o == self.observations[5] and len(observation_set) == 2:
                return self.obs_noise
            else:
                return 1 - self.obs_noise
        else:
            return 0
        
    def observation_function_sampler(self, state, sAct):
        observation_set = self.observation_function(state, sAct)
        if len(observation_set) == 1:
            return self.observations[5]
        else:
            return choices(observation_set, [1 - self.obs_noise, self.obs_noise], k=1)[0]

    def get_reward(self, state, F):
        if state in F:
            return 1
        else:
            return 0

    def pi_theta(self, s, a, theta):
        """
        :param m: the index of a finite sequence of observation, corresponding to K-step memory
        :param a: the sensing action to be given
        :param theta: the policy parameter, the size state_size^3 * sensing_action_size
        :return: the Gibbs policy given the finite memory
        """
        e_x = np.exp(theta[s, :] - np.max(theta[s, :]))
        return (e_x / e_x.sum(axis=0))[a]

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

    def softmax_policy(self, opt_values, goal, tau=0.01, gamma=0.8):
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

    def extract_opt_theta(self, opt_values, F, tau=0.01):
        pi_star = self.softmax_policy(opt_values, F, tau)
        theta = np.log(pi_star)
        return theta

    def get_optimal_policy(self, opt_values, F):
        pi_star = np.zeros([self.state_size, self.action_size])
        theta = self.extract_opt_theta(opt_values, F)
        for s in range(self.state_size):
            for a in range(self.action_size):
                pi_star[s, a] = self.pi_theta(s, a, theta)
        return pi_star
