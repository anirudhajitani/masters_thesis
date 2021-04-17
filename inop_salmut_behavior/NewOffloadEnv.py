# -*- coding: utf-8 -*-


import random
import json
import gym
import time
from gym import spaces
import numpy as np
import statistics

class OffloadEnv(gym.Env):

    """A offloading environment for OpenAI gym"""

    def __init__(self, eval_, lambd, offload, overload, holding, reward, N, seed, env_name=None, folder=None, start_iter=0, step=0, add_stats=True):

        super(OffloadEnv, self).__init__()
        #self.action_space = 2
        self.action_space = spaces.Discrete(2)
        self.eval_ = eval_
        #self.observation_space = 2
        self.seed_train = seed
        np.random.seed(seed)
        random.seed(seed)
        self.observation_high = np.array([20, 20])
        self.observation_low = np.array([0, 0])
        self.observation_space = spaces.Box(
            self.observation_low, self.observation_high)
        self.curr_buffer = 0
        self.max_buffer = 20
        self.cpu_util = 0.0
        self.start_loop = start_iter
        self.end_loop = start_iter + step
        self.start_time = time.time()
        self.current_step = 0
        self.add_stats = add_stats
        self.N = N
        self.prob_r = 0.0
        self.prob_r_val = [0] * 2
        self.c = 2
        self.l = lambd
        self.env_name = env_name
        self.folder = folder
        self.lambd = [lambd] * self.N
        self.mu = 6.0
        self.seed_count = 0
        self.offload_cost = offload
        self.overload_cost = overload
        self.holding_cost = holding
        self.reward = reward
        self.outer_seed_count = 0
        self.overload_count = 0
        self.overload_state = []
        self.med_overload = []
        self.overload_run = []
        self.offload_count = 0
        self.offload_state = []
        self.med_offload = []
        self.offload_run = []
        self.buffer_history = []
        self.cpu_util_history = []
        self.med_buffer = []
        self.med_cpu_util = []
        self.arrivals = 0
        self.departures = 0
        self.arr_count = 0
        self.dept_count = 0
        # Load fixed Event Trajectory
        self.event_traj = np.load("./event_traj.npy")
    """
    For now we assume lambda also evolves at same time for all clients
    """
    def get_lambda_estimates(self):
        print ("ARR, DEP ", self.arrivals, self.departures)
        if self.departures != 0:
            est_lambd = (float(self.arrivals) / float(self.departures)) * (self.mu * self.c)
        else:
            return 2.0
        self.arrivals = 0
        self.departures = 0
        return est_lambd
    
    def get_prob_resources(self):
        print ("Resource 1, 2 ", self.prob_r_val[0], self.prob_r_val[1])
        if self.prob_r_val[1] == 0:
            return 1.0
        p = float(self.prob_r_val[0]) / float(self.prob_r_val[0] + self.prob_r_val[1])
        self.prob_r_val = [0] * 2
        return p 

    def set_lambd(self, lambd):
        self.lambd = lambd

    def set_N(self, N, lambd):
        self.N = N
        self.lambd = lambd

    def get_lambd(self):
        return self.lambd

    def get_N(self):
        return self.N
    
    """
    def get_cpu_util2(self):
        x = np.random.normal(4, 0.25, 1)[0]
        v = min(15.0 + (self.curr_buffer * x), 100.0)
        return int(v/5)
    """
    
    def get_cpu_util(self, ev_type, action):
        if ev_type == 1:
            prob = 0.6
        elif action == 0:
            prob = 0.6
        else:
            prob = 0.6
        prob = 1.0 - prob
        val = np.random.binomial(n=1, p=prob, size=1)[0]
        if ev_type == 1:
            return max(self.cpu_util - val - 1, 0)
        elif action == 0:
            return min(self.cpu_util + val + 1, 20)
        else:
            return min(self.cpu_util - val, 0)

    def get_reward(self):
        # Also adds the state (overload or no) to list
        if self.cpu_util >= 18:
            self.overload_count += 1
            self.overload_state.append(1)
            return -self.overload_cost
        elif self.cpu_util >=6:
            self.overload_state.append(0)
            return self.reward
        else:
            self.overload_state.append(0)
            return 0.0

    def get_prob(self):
        if self.curr_buffer == 0:
            prob = 1.0
        elif self.curr_buffer <= self.c:
            prob = sum(self.lambd) / (sum(self.lambd) +
                                      self.curr_buffer * self.mu)
        else:
            prob = sum(self.lambd) / (sum(self.lambd) + self.c * self.mu)
        return 1.0 - prob

    def _next_observation(self):
        return np.array([self.curr_buffer, self.cpu_util])

    def _take_action(self, action, event_type):
        if event_type == 1:
            self.curr_buffer = max(self.curr_buffer - 1, 0)
        elif action == 0 and event_type == 0:
            self.curr_buffer = min(self.curr_buffer + 1, 20)

    def step(self, action):
        # Execute one time step within the environment
        prob = self.get_prob()
        reward = 0.0
        # Compute probability of event and if greater than random no
        # in event trajectory (event is departure, else arrival
        if prob > self.event_traj[int(self.current_step)]:
            event_type = 1
            self.dept_count += 1
        else:
            event_type = 0
            self.arr_count += 1
        
        self.current_step += 1
        if self.current_step % 1000 == 0:
            print("ARRIVAL COUNT DEPT COUNT STEP ", self.arr_count, self.dept_count, self.current_step)
            self.arr_count = 0
            self.dept_count = 0
        #print ("Event ", event_type, self.current_step)
        # Condition where max_buff len is reached
        if self.curr_buffer == self.max_buffer and event_type == 0 and action == 0:
            reward -= self.overload_cost
        old_cpu = self.cpu_util
        # Also storing buffer history and CPU Utilization history
        self.buffer_history.append(self.curr_buffer)
        self.cpu_util_history.append(self.cpu_util)
        self._take_action(action, event_type)
        self.cpu_util = self.get_cpu_util(event_type, action)
        diff = abs(old_cpu - self.cpu_util)
        if event_type == 0 and action == 0:
            if diff == 1:
                self.prob_r_val[0] += 1
            else:
                self.prob_r_val[1] += 1
        reward += self.get_reward()
        if action == 1:
            self.offload_count += 1
            self.offload_state.append(1)
            reward -= self.offload_cost
            if self.cpu_util < 3:
                reward -= 10
        else:
            self.offload_state.append(0)
        reward -= self.holding_cost * \
            (self.curr_buffer - self.c) if self.curr_buffer - self.c > 0 else 0
        if self.curr_buffer >= self.c:
            if event_type == 0:
                self.arrivals += 1
            else:
                self.departures += 1
        #print("Buffer, CPU Util, Reward, Action, Event = ",
        #      self.curr_buffer, self.cpu_util, reward, action, event_type)
        done = False
        #if self.current_step % 1000 == 999:
        # Saving buffers whenever 10^3 iterations are done 
        if self.current_step % 1000 == 999:
            #print ("Saving overload stats")
            #print ("Overload, Offload count = ", self.overload_count, self.offload_count)
            print ("ARR, DEP ", self.arrivals, self.departures)
            #self.med_offload.append(self.offload_count) 
            #self.med_overload.append(self.overload_count)
            self.med_offload.append(int(self.offload_count)) 
            self.med_overload.append(int(self.overload_count))
            self.med_buffer.append(statistics.median(self.buffer_history))
            self.med_cpu_util.append(statistics.median(self.cpu_util_history))
            self.buffer_history = []
            self.cpu_util_history = []
            self.offload_count = 0
            self.overload_count = 0
            np.save(f"./{self.folder}/results/offload_state_{self.env_name}_{self.seed_train}.npy", self.offload_state)
            np.save(f"./{self.folder}/results/overload_state_{self.env_name}_{self.seed_train}.npy", self.overload_state)
            np.save(f"./{self.folder}/results/offload_med_{self.env_name}_{self.seed_train}.npy", self.med_offload)
            np.save(f"./{self.folder}/results/overload_med_{self.env_name}_{self.seed_train}.npy", self.med_overload)
            np.save(f"./{self.folder}/results/med_curr_buffer_{self.env_name}_{self.seed_train}.npy", self.med_buffer)
            np.save(f"./{self.folder}/results/med_cpu_util_{self.env_name}_{self.seed_train}.npy", self.med_cpu_util)
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        #print ("RESET CALLED")
        #self.buffer_size = 10
        if self.eval_ == True:
            np.random.seed(int(self.seed_count))
            random.seed(int(self.seed_count))
            self.seed_count = (self.seed_count + 1) % 100
        self.curr_buffer = 10 
        self.overload_count = 0
        self.offload_count = 0
        self.cpu_util = 10
        self.current_step = 0
        #print("RESET ", self.outer_seed_count, self.seed_count, " State (Buff, CPU) ", self.curr_buffer, self.cpu_util)
        return self._next_observation()
