import copy
import numpy as np
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


# Used for Atari
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
            is_atari,
            num_actions,
            state_dim,
            device,
            BCQ_threshold=0.3,
            discount=0.999,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
            initial_eps=1,
            end_eps=0.001,
            eps_decay_period=25e4,
            eval_eps=0.001,
            threshold_cpu=17,
            threshold_req=17,
            step_size_slow=0.0075,
            step_size_fast=0.075,
            fixed_state=175,
            T=2,
    ):

        self.device = device

        # buff * cpu * type - 21 * 11 * 2
        self.state_counts = np.ones((441,2), dtype=int)
        self.val_fn = np.zeros((441,2), dtype=float)
        # self.cpu_thres = np.full((231), threshold_cpu, dtype=float)
        self.req_thres = np.full((21), threshold_req, dtype=float)

        # Determine network type
        # self.Q = Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
        # self.Q_target = copy.deepcopy(self.Q)
        # self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount
        self.thres_vec = []
        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.bcq_threshold = BCQ_threshold
        self.threshold_cpu = threshold_cpu
        self.threshold_req = threshold_req

        # Number of training iterations
        self.iterations = 0
        self.slow_iter = 10000
        self.fast_iter = 10
        self.fixed_state = fixed_state
        # LR parameters
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

    def set_threshold_vec(self, thres, val_fn, state_counts):
        self.req_thres = thres
        self.val_fn = val_fn
        self.state_counts = state_counts
        print("Thres vec : ", self.req_thres)
        print("Val fn : ", self.val_fn)

    def encode(self, state):
        # return (state[0] * 11) + (state[1]) + (state[2] * 220)
        return (state[0] * 21) + (state[1])

    def adam_lr(self, state_val, g, step_size, t):
        # g -ve coz we actually want to do plus
        g = -g
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * g
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.power(g, 2)
        m_hat = self.m / (1 - np.power(self.beta_1, t))
        v_hat = self.v / (1 - np.power(self.beta_2, t))
        state_val = state_val - step_size * \
            m_hat / (np.sqrt(v_hat) + self.epsilon)
        #print ("Adam LR ", state_val)
        return state_val

    def adam_lr_thres(self, state_val, g, step_size, t):
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
        # x = np.zeros((3))
        x = np.zeros((2))
        # val = state / 220
        # x[2] = val
        # if val == 1:
        #	state = state - 231
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
        en_state = int(self.encode(state))
        if np.random.uniform(0, 1) > self.eval_eps or eval_ == True:
            action = self.sigmoid_fn(state)
        else:
            action = np.random.randint(self.num_actions)
        self.state_counts[en_state, action] += 1
        if debug:
            print("ACTION : ", action)
        return action

    """
	def select_action(self, state, eval=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > self.eval_eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				q, imt, i = self.Q(state)
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.bcq_threshold).float()
				# Use large negative number to mask actions from argmax
				return int((imt * q + (1. - imt) * -1e8).argmax(1))
		else:
			return np.random.randint(self.num_actions)
	"""

    def projection(self, cpu_state):
        req_thres = self.req_thres[cpu_state]
        for i in range(cpu_state, 21):
            if self.req_thres[i] > req_thres:
                self.req_thres[i] = req_thres

    """
	TODO
	1. Need to figure out rho values updation (fast and slow timescale)
	2. Mellowmax Softmax Function
	3. Threshold Update
	4. Projection Method
	5. Cost Incurred
	"""

    def return_val(self, cpu):
        if random.random() < 0.5:
            return self.req_thres[min(cpu+1, 10)]
        else:
            return -self.req_thres[cpu]

    # def train(self, state, action, rewards, next_state, eval_freq, env_name, folder, j):
    def train(self, replay_buffer, env_name, folder, j, run, eval_freq=1000):
        states, actions, next_states, rewards, done = replay_buffer.sample()
        states = states.numpy()
        actions = actions.numpy()
        next_states = next_states.numpy()
        rewards = rewards.numpy()
        done = done.numpy()
        """
	If file present, load threshold vectors
	"""
        for i in range(replay_buffer.ptr):
            en_state = int(self.encode(states[i]))
            en_next_state = int(self.encode(next_states[i]))
            state = states[i]
            action = actions[i]
            #print ("encode state : ", en_state, state)
            next_state = next_states[i]
            state[0] = int(state[0])
            state[1] = int(state[1])
            next_state[0] = int(next_state[0])
            next_state[1] = int(next_state[1])
            reward = rewards[i]
            T = 0.5
            debug = 0
            self.state_counts[en_state, action] += 1
            #print ("ACTION", action)
            self.iterations += 1
            if debug:
                print("Value fn before : ", en_state, self.val_fn[en_state, action])
            # Updating value function
            g = reward + self.discount * max(self.val_fn[en_next_state, 0], self.val_fn[en_next_state, 1]) \
                - self.val_fn[en_state, action] 
            val_update = self.adam_lr(
                self.val_fn[en_state, action], g, self.step_size_fast, self.state_counts[en_state, action])
            #self.val_fn[en_state, action] -= val_update
            self.val_fn[en_state, action] = val_update
            # else:
            #    self.val_fn[en_state, action] += val_update
            # self.val_fn[en_state] = (1 - rho) * self.val_fn[en_state] + rho * (
            #    reward + self.discount * self.val_fn[en_next_state] - self.val_fn[198])
            if debug:
                print("Value fn after, Action, g", en_state,
                      self.val_fn[en_state, action], action, g)

            # Updating threshold values mul is derivative of sigmoid function
            # Should ideally multiply by -1, hence make g as -ve
            mul = (math.exp((state[1]-self.req_thres[int(state[0])])/self.T)/self.T) / \
                math.pow(
                        (1+math.exp((state[1]-self.req_thres[int(state[0])])/self.T)), 2)
            alpha = np.random.binomial(n=1, p=0.5, size=1)[0]
            if debug:
                print("Thres before : , mul",
                      state[1], self.req_thres[int(state[0])], mul)
            #print ("derivate sigmoid, total val", mul, mul * rho1 * math.pow(-1, alpha) * self.val_fn[en_state])
            #g = mul * math.pow(-1, alpha) * self.val_fn[en_next_state]
            # didnt make mul -ve as math.pow is supposed to be +ve for alpha = 1 but here its opposite
            #g = 1 * mul * math.pow(-1, action) * (reward + self.discount * max(self.val_fn[en_next_state, 0], self.val_fn[en_next_state, 1]))
            #g = math.pow(-1, 1-action) * mul * (reward + self.discount * self.val_fn[en_next_state]) 
            g = math.pow(-1, 1-action) * mul * (reward + self.discount * max(self.val_fn[en_next_state, 0], self.val_fn[en_next_state, 1]))
            # if g > 0:
            #    print ("POSITIVE ", action)
            # else:
            #    print ("-VE ", action)
            #print ("Thres before ", self.req_thres[int(state[1])])
            #self.req_thres[int(state[0])] = min(max(self.req_thres[int(state[0])] + math.pow(-1, action) * self.adam_lr_thres(self.req_thres[int(state[0])], g, self.step_size_slow, self.state_counts[en_state, action]), 0.0), 20.0)
            self.req_thres[int(state[0])] = min(max(self.adam_lr(self.req_thres[int(
                state[0])], g, self.step_size_slow, self.state_counts[en_state, action]), 0.0), 20.0)
            if debug:
                print("Thres after ", self.req_thres[int(
                    state[0])], " Action ", action, " g = ", g)
            #print("Thres after : ", state[1], self.req_thres[int(state[1])])
            # Projection operation
            #self.projection(int(state[0]))

            # self.maybe_update_target()
        print("CPU thres", self.req_thres)
        self.thres_vec.append(list(self.req_thres))
        #print("CPU thres vec", self.thres_vec)
        # print(f"./{folder}/buffers/{env_name}_thresvec_{j}.npy")
        np.save(
            f"./{folder}/buffers/thresvec_{run}_{env_name}_{j}.npy", self.req_thres)
        np.save(
            f"./{folder}/buffers/val_fn_{run}_{env_name}_{j}.npy", self.val_fn)
        np.save(
            f"./{folder}/buffers/state_counts_{run}_{env_name}_{j}.npy", self.state_counts)
        #print("VAL", self.val_fn)
        #print("State counts :", self.state_counts)

    """	
	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()

		# Compute the target Q value
		with torch.no_grad():
			q, imt, i = self.Q(next_state)
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

			# Use large negative number to mask actions from argmax
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

			q, imt, i = self.Q_target(next_state)
			target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

		# Get current Q estimate
		current_Q, imt, i = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		i_loss = F.nll_loss(imt, action.reshape(-1))

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()
		# print (" Loss ", q_loss, i_loss)
		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


    def train_replay(self, replay_buffer):

        states, action, next_states, reward, done = replay_buffer.sample()
        states = states.numpy()
        action = action.numpy()
        next_states = next_states.numpy()
        reward = reward.numpy()
        done = done.numpy()
        for i in range(len(action)):
            en_state = int(self.encode(states[i]))
            en_next_state = int(self.encode(next_states[i]))
            state = states[i]
            # print ("encode state : ", en_state, state)
            next_state = next_states[i]
            state[0] = int(state[0])
            state[1] = int(state[1])
            next_state[0] = int(next_state[0])
            next_state[1] = int(next_state[1])
            # print ("encode next state : ", en_next_state)
            # self.state_counts[en_state] += 1
            # rho = math.pow(math.floor(fast_iter*0.01)+2,-0.6)
            rho = math.pow(math.floor(
                self.state_counts[en_state]*0.01)+2, -0.4)
            # rho1 = 10*math.pow(slow_iter,-1)
            # rho1 = 10*math.pow(self.state_counts[en_state], -1)
            rho1 = math.pow(self.state_counts[en_state], -0.6)
            # print("RHO ", rho, rho1)
            next_action = self.select_action(next_state)
            # print("Value fn before : ", en_state, self.val_fn[en_state])
            # Updating value function
            self.val_fn[en_state] = (1 - rho) * self.val_fn[en_state] + rho * (
                reward[i] + self.discount * self.val_fn[en_next_state] - self.val_fn[0])
            # print("Value fn after : ", en_state, self.val_fn[en_state])

            # Updating threshold values mul is derivative of sigmoid function
            mul = (math.exp(state[0]-self.req_thres[int(state[1])]/0.5)/0.5) / \
                math.pow(
                    (1+math.exp(state[0]-self.req_thres[int(state[1])])), 2.0)
            alpha = np.random.binomial(n=1, p=0.5, size=1)[0]
            # print("Thres before : ", state[1], self.req_thres[int(state[1])])
            self.req_thres[int(state[1])] = min(max(self.req_thres[int(state[1])] - mul *
                                                    rho1 * math.pow(-1, alpha) * self.val_fn[en_state], 0.0), 20.0)
            # print("Thres after : ", state[1], self.req_thres[int(state[1])])
            # Projection operation
            self.projection(int(state[1]))

            self.iterations += 1
            self.slow_iter += 1
            self.fast_iter += 1
        # self.maybe_update_target()
        print("CPU thres", self.req_thres)
        print("VAL", self.val_fn)
    """
