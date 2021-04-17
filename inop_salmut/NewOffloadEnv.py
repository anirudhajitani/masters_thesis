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
        # Action space
        self.action_space = spaces.Discrete(2)
        self.eval_ = eval_
        np.random.seed(seed)
        random.seed(seed)
        # State space 21*21 = 441
        self.observation_high = np.array([20, 20])
        self.observation_low = np.array([0, 0])
        self.observation_space = spaces.Box(
            self.observation_low, self.observation_high)
        # Intialize buffers
        self.curr_buffer = 0
        self.max_buffer = 20
        self.cpu_util = 0.0
        # Initialize start and end loop if passed
        self.start_loop = start_iter
        self.end_loop = start_iter + step

        # Initialize times
        self.start_time = time.time()
        self.current_step = 0
        # Needed for Parameter estimation
        self.add_stats = add_stats
        self.N = N
        # Estimating probability of resources
        self.prob_r = 0.0
        self.prob_r_val = [0] * 2
        # Cores
        self.c = 2
        self.l = lambd
        self.env_name = env_name
        self.folder = folder
        self.lambd = [lambd] * self.N
        self.mu = 6.0
        self.seed_count = 0
        # Costs and rewards
        self.offload_cost = offload
        self.overload_cost = overload
        self.holding_cost = holding
        self.reward = reward

        self.outer_seed_count = 0
        # Behavioral Analysis
        self.overload_count = 0
        self.overload_state = []
        self.med_overload = []
        self.overload_run = []
        self.offload_count = 0
        self.offload_state = []
        self.med_offload = []
        self.offload_run = []

        # Count arrivals and departures
        self.arrivals = 0
        self.departures = 0
    """
    For now we assume lambda also evolves at same time for all clients
    """

    def get_lambda_estimates(self):
        """
        Parameter Estimation -> estimate lambda from client arrivals and
        departures in the period, and then reset the value of arrivals
        and departures.
        """
        print("ARR, DEP ", self.arrivals, self.departures)
        if self.departures != 0:
            est_lambd = (float(self.arrivals) /
                         float(self.departures)) * (self.mu * self.c)
        else:
            return 2.0
        self.arrivals = 0
        self.departures = 0
        return est_lambd

    def get_prob_resources(self):
        """
        Estimate probability of resources required based on historic count
        of request required for resources.
        """
        print("Resource 1, 2 ", self.prob_r_val[0], self.prob_r_val[1])
        if self.prob_r_val[1] == 0:
            return 1.0
        p = float(self.prob_r_val[0]) / \
                  float(self.prob_r_val[0] + self.prob_r_val[1])
        self.prob_r_val = [0] * 2
        return p

    def set_lambd(self, lambd):
        """
        Set lambda for the environment
        """
        self.lambd = lambd

    def set_N(self, N, lambd):
        """
        Set N for the environment
        """
        self.N = N
        self.lambd = lambd

    def get_lambd(self):
        return self.lambd

    def get_N(self):
        return self.N

    def get_cpu_util2(self):
        """
        Used earlier when states evolved according to function.
        Now it is using MDP.
        """
        x = np.random.normal(4, 0.25, 1)[0]
        v = min(15.0 + (self.curr_buffer * x), 100.0)
        return int(v/5)

    def get_cpu_util(self, ev_type, action):
        """
        CPU load MDP evolution based on event type and action
        For now assume all transitions are 0.6 to one l+1 and
        0.4 for l+2.
        """
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
        """
        Return overload cost incurred by the system
        """
        if self.cpu_util >= 18:
            if self.eval_ == True:
                self.overload_count += 1
            return -self.overload_cost
        elif self.cpu_util >= 6:
            return self.reward
        else:
            return 0.0

    def get_prob(self):
        """
        Get probability of new event (arrival/departure)
        """
        if self.curr_buffer == 0:
            prob = 1.0
        elif self.curr_buffer <= self.c:
            prob = sum(self.lambd) / (sum(self.lambd) +
                                      self.curr_buffer * self.mu)
        else:
            prob = sum(self.lambd) / (sum(self.lambd) + self.c * self.mu)
        return 1.0 - prob

    def _next_observation(self):
        """
        Return next action
        """
        return np.array([self.curr_buffer, self.cpu_util])

    def _take_action(self, action, event_type):
        """
        Updating the buffer after taking action
        """
        if event_type == 1:
            self.curr_buffer = max(self.curr_buffer - 1, 0)
        elif action == 0 and event_type == 0:
            self.curr_buffer = min(self.curr_buffer + 1, 20)

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        # Get probability of event
        prob = self.get_prob()
        reward = 0.0
        # Generate event type
        event_type = np.random.binomial(n=1, p=prob, size=1)[0]
        # Condition where max_buff len is reached
        # Provide a high negative cost when max buffer reached and accept action
        # It is similar to rejecting the request
        if self.curr_buffer == self.max_buffer and event_type == 0 and action == 0:
            reward -= self.overload_cost
        # Get old load
        old_cpu = self.cpu_util
        # Update buffer size
        self._take_action(action, event_type)
        # Update CPU Load
        self.cpu_util = self.get_cpu_util(event_type, action)

        # Update resource required based on observed CPU differences
        # Used for parameter estimation only
        diff = abs(old_cpu - self.cpu_util)
        if event_type == 0 and action == 0:
            if diff == 1:
                self.prob_r_val[0] += 1
            else:
                self.prob_r_val[1] += 1
        # Get overload cost
        reward += self.get_reward()

        # Update costs with other costs
        if action == 1:
            if self.eval_ == True:
                self.offload_count += 1
            reward -= self.offload_cost
            # High cost when underitilized and still overloads
            if self.cpu_util < 3:
                reward -= 10
        # Add holding cost
        reward -= self.holding_cost * \
            (self.curr_buffer - self.c) if self.curr_buffer - self.c > 0 else 0

        # Update arrival and departure counts
        if self.curr_buffer >= self.c:
            if event_type == 0:
                self.arrivals += 1
            else:
                self.departures += 1
        # print("Buffer, CPU Util, Reward, Action, Event = ",
        #      self.curr_buffer, self.cpu_util, reward, action, event_type)
        done = False
        """
        if self.current_step % 1000 == 0:
            # print ("Episode complete")
            done = True
            self.overload_state.append(self.overload_count)
            # print ("Overload count = ", self.overload_count)
        """
        # Get next state
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        # self.buffer_size = 10
        """
        For evaluations we update the overload and offload counts.
        Get these metrics and compute median, quartile ranges
        """
        if self.eval_ == True:
            # print ("SEED COUNT", self.seed_count)
            if self.seed_count > 0 and self.add_stats == True:
                self.overload_state.append(self.overload_count)
                self.offload_state.append(self.offload_count)
            if self.seed_count % 100 == 99 and self.add_stats == True:
                self.overload_run.append(statistics.mean(self.overload_state))
                # self.std_overload.append(statistics.stdev(self.overload_state))
                self.overload_state = []
                self.offload_run.append(statistics.mean(self.offload_state))
                # self.std_offload.append(statistics.stdev(self.offload_state))
                self.offload_state = []
                self.outer_seed_count = (self.outer_seed_count + 1) % 10
                # print ("AVG OVERLOAD", self.avg_overload, self.std_overload, self.env_name)
                # print ("AVG Offload", self.avg_offload, self.std_offload, self.env_name)

                # Once we run for 10 different seeds of trainings
                if self.outer_seed_count % 10 == 0: 
                    self.med_overload.append(np.percentile(self.overload_run, [25,50,75]))
                    self.med_offload.append(np.percentile(self.offload_run, [25,50,75]))
                    print ("Overload, Offload", np.percentile(self.overload_run, [25,50,75]), 
                            np.percentile(self.offload_run, [25,50,75]))
                    # print ("SAVING OVERLOAD COUNT")
                    self.overload_run = []
                    self.offload_run = []
                    # Saving the overload and offload metrics
                    if self.start_loop != self.end_loop:
                        np.save(f"./{self.folder}/results/overload_med_{self.start_loop}_{self.end_loop}_{self.env_name}.npy", self.med_overload) 
                        # np.save(f"./{self.folder}/results/overload_std_{self.env_name}.npy", self.std_overload) 
                        np.save(f"./{self.folder}/results/offload_med_{self.start_loop}_{self.end_loop}_{self.env_name}.npy", self.med_offload) 
                        # np.save(f"./{self.folder}/results/offload_std_{self.env_name}.npy", self.std_offload) 
            np.random.seed(int(self.seed_count))
            random.seed(int(self.seed_count))
            self.seed_count = (self.seed_count + 1) % 100
        # self.curr_buffer = random.randrange(10, 20)
        # Intialize the states and counts to values (State Reset performed)
        self.curr_buffer = 10 
        self.overload_count = 0
        self.offload_count = 0
        self.cpu_util = 10
        # print("RESET ", self.outer_seed_count, self.seed_count, " State (Buff, CPU) ", self.curr_buffer, self.cpu_util)
        return self._next_observation()
