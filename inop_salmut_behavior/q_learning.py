import copy
import numpy as np
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


# We are not using these (in case we need Function approximator
class Conv_Q(nn.Module):
    def __init__(self, frames, num_actions):
        super(Conv_Q, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.q1 = nn.Linear(3136, 512)
        self.q2 = nn.Linear(512, num_actions)

        self.i1 = nn.Linear(3136, 512)
        self.i2 = nn.Linear(512, num_actions)

    def forward(self, state):
        c = F.relu(self.c1(state))
        c = F.relu(self.c2(c))
        c = F.relu(self.c3(c))

        q = F.relu(self.q1(c.reshape(-1, 3136)))
        i = F.relu(self.i1(c.reshape(-1, 3136)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i


# Used for Box2D / Toy problems
class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, 64)
        self.q2 = nn.Linear(64, 64)
        self.q3 = nn.Linear(64, num_actions)

        self.i1 = nn.Linear(state_dim, 64)
        self.i2 = nn.Linear(64, 64)
        self.i3 = nn.Linear(64, num_actions)

    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i


class structured_learning(object):
    def __init__(
            self,
            num_actions,
            state_dim,
            #device,
            discount=0.95,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            eval_eps=0.001,
            threshold_cpu=17,
            threshold_req=17,
            step_size_slow=0.002,
            step_size_fast=0.03,
            #Best one till now
            #step_size_slow = 0.0005,
            #step_size_fast = 0.0075,
            #step_size_slow = 0.0005,
            #step_size_fast = 0.0075,
            #step_size_slow = 0.0001,
            #step_size_fast = 0.0005,
            # For average MDP case
            fixed_state=175,
            # Temperature variable
            T=2,
    ):

        #self.device = device

        # Initialize state counts and value functions and request thresholds
        self.state_counts = np.ones((441,2), dtype=int)
        self.val_fn = np.zeros((441,2), dtype=float)
        #self.req_thres = np.full((21), threshold_req, dtype=float)

        # Determine network type
        # self.Q = Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
        # self.Q_target = copy.deepcopy(self.Q)
        # self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount
        self.thres_vec = []

        # Decay for eps
        """
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
        
        # Evaluation hyper-parameters
        self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
        """
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold_cpu = threshold_cpu
        self.threshold_req = threshold_req

        # Number of training iterations
        self.iterations = 0
        self.slow_iter = 10000
        self.fast_iter = 10
        self.fixed_state = fixed_state
        # LR parameters for ADAM optimizer
        self.m = 0.0
        self.v = 0.0
        self.T = T
        self.epsilon = 1e-6
        self.step_size_slow = step_size_slow
        self.step_size_fast = step_size_fast
        self.beta_1 = 0.9
        self.beta_2 = 0.99

        print("SLOW LR, FAST LR, FIXED STATE ", self.step_size_slow,
              self.step_size_fast, self.fixed_state)

    def set_threshold_vec(self, thres):
        # Set threshold vector (aka policy)
        self.req_thres = thres
        #print ("Thres vec : ", self.req_thres)

    def set_val_fn(self, val_fn):
        # Set threshold vector (aka policy)
        self.val_fn = val_fn
        #print ("Thres vec : ", self.req_thres)
    
    def encode(self, state):
        # State encoding
        return (state[0] * 21) + (state[1])

    def adam_lr(self, state_val, g, step_size, t):
        # ADAM optimizer updating parameters and LR
        # g -ve coz we actually want to do plus
        g = -g
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * g
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(g, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, t))
        v_hat = self.v / (1 - np.power(self.beta_2, t))
        state_val = state_val - step_size * \
            m_hat / (np.sqrt(v_hat) + self.epsilon)
        return state_val

    def adam_lr_thres(self, state_val, g, step_size, t):
        # ADAM optimizer updating parameters and LR
        # g -ve coz we actually want to do plus
        g = -g
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * g
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(g, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, t))
        v_hat = self.v / (1 - np.power(self.beta_2, t))
        val_update = step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
        #print ("Change ", val_update)
        return step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # return state_val

    def decode(self, state):
        # Decode the encoded state
        x = np.zeros((2))
        val = state / 21
        x[0] = val
        x[1] = state % 21
        return x

    def sigmoid_fn(self, state, debug=0):
        # Simple as of now
        """
        If this value is > 0.5 i.e. state[0] - self.req_thres > 0, then we need to offload, return 1 
        will high probability as a=1 for offload
        """
        # print ("State 1 ", state[1], type(state[1]))
        prob = math.exp((state[1] - self.req_thres[int(state[0])])/self.T) / \
            (1 + math.exp((state[1] - self.req_thres[int(state[0])])/self.T))
        if debug:
            print("Sigmoid  state, threshold, prob ",
                  state, self.req_thres[int(state[1])], prob)
        return np.random.binomial(n=1, p=prob, size=1)

    def select_action(self, state, eval_=False, debug=0):
        #print (state)
        """
        Action selection method \eps-greedy
        """
        en_state = int(self.encode(state))
        if np.random.uniform(0, 1) > self.eval_eps or eval_ == True:
            action = np.argmax(self.val_fn[en_state]) 
        else:
            action = np.random.randint(self.num_actions)
        self.state_counts[en_state, action] += 1
        if debug:
            print("ACTION : ", action)
        return action

    def projection(self, cpu_state):
        """
        Projection operator as described in paper
        """
        req_thres = self.req_thres[cpu_state]
        for i in range(cpu_state, 21):
            if self.req_thres[i] > req_thres:
                self.req_thres[i] = req_thres

    def train(self, state, action, reward, next_state, eval_freq, env_name, folder, j, i):
        """
        SALMUT Training method
        """
        # Encode state and next-state
        en_state = int(self.encode(state))
        en_next_state = int(self.encode(next_state))
        state[0] = int(state[0])
        state[1] = int(state[1])
        action = int(action)
        #T = 0.5
        # Set to 1 to for detailed logging
        debug = 0
        next_state[0] = int(next_state[0])
        next_state[1] = int(next_state[1])
        # print ("encode next state : ", en_next_state)
        # self.state_counts[en_state] += 1
        self.iterations += 1

        # Earlier trying out different learning rates (now using ADAM)
        # rho = math.pow(math.floor(
        #    self.state_counts[en_state]*0.1)+2, -0.65)
        #rho1 = math.pow(self.state_counts[en_state], -1)
        #print("RHO ", rho, rho1)
        # next_action = self.select_action(next_state)

        if debug:
            print("Value fn before : ", en_state, self.val_fn[en_state, action])
        # Updating value function

        # TD Learning Update
        # TD Error
        #g = reward + self.discount * \
        #    max(self.val_fn[en_next_state, 0],
        #        self.val_fn[en_next_state, 1]) - self.val_fn[en_state, action]
        g = reward + self.discount * max(self.val_fn[en_next_state, 0], self.val_fn[en_next_state, 1]) - self.val_fn[en_state, action]
        # Value function update
        #val_update = self.adam_lr(
        #    self.val_fn[en_state, action], g, self.step_size_fast, self.state_counts[en_state, action])
        self.val_fn[en_state, action] = (1 - 0.005) * self.val_fn[en_state, action] + 0.005 * g
        if debug:
            print("Value fn after, Action, g", en_state,
                  self.val_fn[en_state, action], action, g)
        
        """
        # Updating threshold values mul is derivative of sigmoid function
        # Should ideally multiply by -1, hence make g as -ve
        # Derivative of sigmoid function
        mul = (math.exp((state[1]-self.req_thres[int(state[0])])/self.T)/self.T) / \
            math.pow(
                (1+math.exp((state[1]-self.req_thres[int(state[0])])/self.T)), 2)
        alpha = np.random.binomial(n=1, p=0.5, size=1)[0]
        if debug:
            print("Thres before : , mul",
                  state[1], self.req_thres[int(state[0])], mul)
        # Policy Update
        g = math.pow(-1, 1-action) * mul * (reward + self.discount * max(self.val_fn[en_next_state, 0], self.val_fn[en_next_state, 1]))
        #  max(self.val_fn[en_next_state, 0], self.val_fn[en_next_state, 1]))
        # Update threshold (bound in 0 and 20)
        self.req_thres[int(state[0])] = min(max(self.adam_lr(self.req_thres[int(
            state[0])], g, self.step_size_slow, self.state_counts[en_state, action]), 0.0), 20.0)
        if debug:
            print("Thres after ", self.req_thres[int(
                state[0])], " Action ", action, " g = ", g)
        # Projection operation
        #self.projection(int(state[0]))
        """
        # If iterations == eval frequency (save the parameters in disk)
        if self.iterations % eval_freq == 0:
            #print("CPU thres", self.req_thres)
            #self.thres_vec.append(list(self.val_fn).copy())
            #print("CPU thres vec", self.thres_vec)
            # print(f"./{folder}/buffers/{env_name}_thresvec_{j}.npy")
            #np.save(f"./{folder}/buffers/thresvec_{env_name}_{j}.npy", self.thres_vec)
            #print(to_add_val_fn)
            np.save(f"./{folder}/buffers/val_fn_{env_name}_{j}_{i}.npy", self.val_fn)
            #print("VAL", self.val_fn)
            #print("State counts :", self.state_counts)
