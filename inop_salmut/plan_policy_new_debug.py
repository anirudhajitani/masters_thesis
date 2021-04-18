import numpy as np
import sys
from matplotlib import pyplot as plt


class PlanPolicy():
    def __init__(self, N, lambd, overload, offload, holding, reward, gamma=0.95, prob=0.6):
        # Initialize all states, probabilites
        self.states = []
        self.N = N
        self.C = 2
        self.lambd = lambd
        self.mu = 6
        self.overload_cost = overload
        self.offload_cost = offload
        self.holding_cost = holding
        self.reward = reward
        self.prob_1 = prob
        self.prob_2 = prob
        self.prob_3 = prob
        self.gamma = gamma
        # We will encode the state as int 21*21 = 441
        self.N_STATES = 441
        # Initialize value functions and policies
        self.V = np.zeros(self.N_STATES)
        self.policy = [1 for s in range(self.N_STATES)]
        for i in range(self.N_STATES):
            self.states.append(i)
        self.actions = [0, 1]
        self.N_ACTIONS = len(self.actions)
        # Initialize transition probabilites and rewards
        # transition probability
        self.P = np.zeros((self.N_STATES, self.N_ACTIONS, self.N_STATES))
        self.R = np.zeros(
            (self.N_STATES, self.N_ACTIONS, self.N_STATES))  # rewards
        # Sum of all user lambda
        self.lam_name = round(sum(self.lambd), 1)

    def encode(self, state):
        # State encoding
        return state[0] * 21 + state[1]

    def get_prob(self, state):
        # Get probability of arrival or departure based on lambda and mu
        buff = state % 21
        if buff == 0:
            prob = 1.0
        elif buff <= self.C:
            prob = float(sum(self.lambd)) / \
                float((sum(self.lambd) + buff * self.mu))
        else:
            prob = float(sum(self.lambd)) / \
                float((sum(self.lambd) + self.C * self.mu))
        return prob

    def calc_P(self):
        """
        Calculating transition probability matrix
        """
        # prob_1 -> when departure, the probability of resource change
        # prob_2 -> when arrival and accept, the probability of resource change
        # prob_3 -> when arrival and offload, the probability of resource change
        prob_1 = self.prob_1
        prob_2 = self.prob_2
        prob_3 = self.prob_3

        # Departure Event
        #print ("DEPT")
        """
        i - CPU load
        j - buffer length
        """
        for i in range(0, 21):
            for j in range(0, 21):
                prob = self.get_prob(i*21+j)
                state_i = [i, j]
                state_j = [max(i-1, 0), max(j-1, 0)]
                state_k = [max(i-2, 0), max(j-1, 0)]
                self.P[self.encode(state_i), :, self.encode(
                    state_j)] += (1 - prob) * prob_1
                self.P[self.encode(state_i), :, self.encode(
                    state_k)] += (1 - prob) * (1 - prob_1)

        # Transition Probabilities when accept request and arrival event
        #print ("ACCEPT")
        for i in range(0, 21):
            for j in range(0, 21):
                prob = self.get_prob(i*21+j)
                state_i = [i, j]
                state_j = [min(i+1, 20), min(j+1, 20)]
                state_k = [min(i+2, 20), min(j+1, 20)]
                self.P[self.encode(state_i), 0, self.encode(
                    state_j)] += prob * prob_2
                self.P[self.encode(state_i), 0, self.encode(
                    state_k)] += prob * (1 - prob_2)

        # Transition Probabilities when offload request
        # print("OFFLOAD")
        for i in range(0, 21):
            for j in range(0, 21):
                prob = self.get_prob(i*21+j)
                state_i = [i, j]
                state_j = [i, j]
                state_k = [max(i-1, 0), j]
                self.P[self.encode(state_i), 1, self.encode(
                    state_j)] += prob * prob_3
                self.P[self.encode(state_i), 1, self.encode(
                    state_k)] += prob * (1 - prob_3)

        """" 
        print ("Transition Matrix ", self.P)
        for i in range(self.N_STATES):
            for j in range(self.N_ACTIONS):
                print ("SUM ", i, j, np.sum(self.P[i, j, :]))
        """

    def policy_iteration(self):
        """
        Policy Evaluation Algorithm
        """
        is_value_changed = True
        is_policy_stable = False
        iterations = 0
        theta = 0.001

        while is_policy_stable is False:
            is_value_changed = True
            iterations = 0
            # Policy Evaluation
            while is_value_changed:
                iterations += 1
                delta = 0
                for s in range(self.N_STATES):
                    v = self.V[s]
                    self.V[s] = sum([self.P[s, self.policy[s], s1] * (self.R[s, self.policy[s], s1] +
                                                                      self.gamma * self.V[s1]) for s1 in range(self.N_STATES)])
                    delta = max(delta, abs(v - self.V[s]))
                    #print ("Iter = ", iterations, " S = ", s, " v = ", v , " V[s] = ", V[s], " delta = ", delta)
                if delta < theta:
                    is_value_changed = False

            is_policy_stable = True
            # Policy Improvement
            for s in range(self.N_STATES):
                old_action = self.policy[s]
                action_value = np.zeros((self.N_ACTIONS), dtype=float)
                for a in range(self.N_ACTIONS):
                    action_value[a] = sum(
                        [self.P[s, a, s1] * (self.R[s, a, s1] + self.gamma * self.V[s1]) for s1 in range(self.N_STATES)])
                self.policy[s] = np.argmax(action_value)
                if old_action != self.policy[s]:
                    is_policy_stable = False

    def plot_graph(self, decay=False):
        """
        Plotting policy
        """
        policy = np.array(self.policy)
        fig = plt.figure(figsize=(12, 8))
        labels = {0: 'ACCEPT', 1: 'OFFLOAD'}
        #policy = policy.reshape(22,20)
        policy = policy.reshape(21, 21)
        #policy = np.flip(policy, 0)
        im = plt.imshow(policy, interpolation='nearest',
                        cmap='gray_r', origin='lower')
        plt.colorbar(im, label=labels)
        str_1 = 'Overload=' + str(self.overload_cost) + 'Offload=' + str(self.offload_cost) + 'Holding=' + str(
            self.holding_cost) + 'Reward=' + str(self.reward) + 'Prob=' + str(self.prob_1) + 'Lambda=' + str(sum(self.lambd)) \
            + 'Gamma=' + str(self.gamma) + 'Decay=' + str(decay)
        plt.title(str_1)
        print(str_1)
        plt.xlabel('Request Size')
        plt.ylabel('CPU Utilzation')
        plt.legend()
        # plt.show()
        filename = str_1 + '.png'
        # fig.savefig(f"./plan_plots/{filename}")
        fig.savefig(f"./policies_analysis_new/{filename}")

    def compute_policy(self, plot=False, decay_rew=False, save=False, name=None):
        #step = self.overload_cost / 5
        """
        Setting rewards matrix
        """
        # For CPU util between 8 and 18
        self.R[:, :, 126:378] += self.reward
        # For offloading
        self.R[:, 1, :] -= self.offload_cost
        # If fixed overload structure, we use this
        if decay_rew == False:
            self.R[:, :, 378:] -= self.overload_cost
        """
        if decay_rew == True:
            self.R[357:378, :, :] -= (self.overload_cost - step)
            self.R[336:357, :, :] -= (self.overload_cost - (2*step))
            self.R[315:336, :, :] -= (self.overload_cost - (3*step))
            #self.R[294:315, :, :] -= (self.overload_cost - (4*step))
            #self.R[273:294, :, :] -= (self.overload_cost - (5*step))
        """
        # Adding holding cost and other costs
        for i in range(0, 21):
            for j in range(0, 21):
                prob = self.get_prob(i*21+j)
                # If step-size overload cost structure
                if decay_rew == True:
                    if i >= 16:
                        self.R[:, :, i*21 + j] -= (i/2.0)
                if j % 21 > self.C:
                    self.R[:, :, i*21 + j] -= self.holding_cost * \
                        ((j % 21) - self.C)
                if j % 21 == 20:
                    self.R[i*21 + j, 0, :] -= self.overload_cost * prob
                if i < 3:
                    self.R[:, 1, i*21 + j] -= 10
        print("START ", self.overload_cost, self.holding_cost,
              self.reward, self.lam_name, self.prob_1, self.gamma)
        # Calculate P matrix
        self.calc_P()
        # Perform Policy Iteration
        self.policy_iteration()
        # Plot policy
        if plot == True:
            self.plot_graph(decay_rew)
        self.print_policy()
        # Provide policy name if not given
        if name is None:
            str_1 = 'Overload=' + str(self.overload_cost) + 'Offload=' + str(self.offload_cost) + 'Holding=' + str(
                self.holding_cost) + 'Reward=' + str(self.reward) + 'Prob=' + str(self.prob_1) + 'Lambda=' + str(sum(self.lambd)) \
                + 'Gamma=' + str(self.gamma) + 'Decay=' + str(decay_rew)
            name = f"./policies_analysis/policy_plan_lambd_{sum(self.lambd)}.npy"
        # Saving policy
        if save == True:
            print("SAVING POLICY ", name)
            np.save(name, self.policy)
        return self.policy

    def print_policy(self):
        np.save(f"./value_fn/new_value_fn_{self.lam_name}.npy", self.V)
