from flask import Flask, request, redirect
from flask_restful import Resource, Api
import requests
import psutil
import time
import numpy as np
import math
import os
import threading as th
import random
import subprocess

app = Flask(__name__)
api = Api(app)


class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done, episode_done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy",
                self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(
            f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(
            f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(
            f"{save_folder}_next_state.npy")[:self.crt_size]

"""
This is the notification class to get notifications or signals
indicating to perform certain actions..
"""
class Notify (Resource):
    def load_req_thres(self):
        global req_thres
        # Load policy if it exists
        if os.path.exists("./req_thres.npy"):
            req_thres = np.load("./req_thres.npy")
            req_thres = req_thres
            print("New Policy Request threshold : ", req_thres)

    def get(self):
        global buffer
        global run
        global lock
        global overload_count
        global offload_count
        global overload_vec
        global offload_vec
        #print("Notification of offload")
        # Notification from microservice
        notify = request.args.get('offload')
        notify = int(notify)
        # Notify = 0 specifies to load the new policy
        if notify == 0:
            self.load_req_thres()
        elif notify < 0:
            # negative notification sends the current run (seed)
            run = abs(notify)
            print("RUN : ", run)
            random.seed(run)
        else:
            # Positive notification means the end of one mini-run
            # Save all buffers and behavioral counts
            lock.acquire()
            buffer.save('buffer_' + str(run) + '_' + str(notify))
            buffer.ptr = 0
            buffer.crt_size = 0
            overload_vec.append(overload_count)
            offload_vec.append(offload_count)
            #print ("Overload count ", overload_vec)
            #print ("Offload count ", offload_vec)
            np.save(f'buffer_{run}_overload_count.npy', overload_vec)
            np.save(f'buffer_{run}_offload_count.npy', offload_vec)
            overload_count = 0
            offload_count = 0
            lock.release()

"""
This class is used to handle requests and perform admission control
"""
class Greeting (Resource):
    def __init__(self, overload=30.0, offload=1.0, reward=0.2, holding=0.12):
        self.overload = overload
        self.offload = offload
        self.reward = reward
        self.holding = holding
        self.c = 2
        self.T = 0.5
        self.num_actions = 2

    def sigmoid_fn(self, cpu_util, buffer, debug=0):
        global req_thres
        # Simple as of now
        """
        If this value is > 0.5 i.e. state[0] - req_thres > 0, then we need to offload, return 1
        will high probability as a=1 for offload
        """
        prob = math.exp((cpu_util - req_thres[int(buffer)])/self.T) / \
            (1 + math.exp((cpu_util - req_thres[int(buffer)])/self.T))
        if debug:
            print("Sigmoid  state, threshold, prob ",
                  state, req_thres[int(cpu_util)], prob)
        return np.random.binomial(n=1, p=prob, size=1)

    def select_action(self, cpu_util, buffer, eval_=False, debug=0):
        if np.random.uniform(0, 1) > 0.005 or eval_ == True:
            action = self.sigmoid_fn(cpu_util, buffer)
        else:
            action = np.random.randint(self.num_actions)
        if debug:
            print("ACTION : ", action)
        return action

    def get_reward(self, cpu_util, buffer, action, debug=1):
        """
        Compute all costs in this function and return
        """
        global buff_size
        global lock
        global offload_count
        global overload_count
        rew = 0.0
        if action == 1:
            rew -= self.offload
            offload_count += 1
        if cpu_util < 3:
            if action == 1:
                rew -= self.overload
                #print("Low util offload")
        elif cpu_util >= 6 and cpu_util <= 17:
            rew += self.reward
            # print("Reward")
        elif cpu_util >= 18:
            rew -= self.overload
            overload_count += 1
            # print("Overload")
        if buffer == buff_size and action == 0:
            rew -= self.overload
            overload_count += 1
            #print("Buffer Full")
        rew -= self.holding * \
            (buffer - self.c) if buffer - self.c > 0 else 0
        return rew

    def get_load(self, str1):
        global start
        # load = os.popen(
        #    "top -b -n 2 -d 0.5 -p 1 | tail -1 | awk '{print $9}'").read()
        # This shell script will get the load of all process, add them and return the load
        load = os.popen(
            "ps -u root -o %cpu,stat | grep -v 'Z' | awk '{cpu+=$1} END {print cpu}'").read()
        load = float(load)
        #print ("Load ", load, str1)
        return min(int(load/5), 20)

    def get(self):
        global buff_len
        global buff_size
        global start_time
        global offload
        global buffer
        global file_count
        global load
        global lock
        global run
        count = request.args.get('count')
        #print ("Main function 1 Lock status", lock.locked())
        lock.acquire()
        # Get current load
        load = self.get_load("new arrival")
        # Get previous state and store in prev_state
        prev_state = [buff_len, load]
        # Get action
        action = self.select_action(load, buff_len)
        if action == 0:
            buff_len = min(buff_len + 1, 20)
        rew = self.get_reward(load, buff_len, action)
        # print("ARRIVAL State, Action Reward",
        #        prev_state, action, rew)
        
        # Add the state, action, next_state as 0 (will process them later), reward, dones etc
        buffer.add(prev_state, action, [0, 0], rew, 0, 0, 0)

        # If buffer is full save the buffer along run and file count information
        if buffer.ptr == buffer.max_size - 1:
            file_count += 1
            buffer.save('buffer_' + str(run) + '_' + str(file_count))
        lock.release()
        if action == 0:
            # Perform task
            #count = int(count)
            # One may need to change this parameter by monitoring the load
            t = random.expovariate(0.13)
            # Cap each request to be 20s
            t = min(t, 20.0)
            for i in range(1):
                # Execute application (run a workload) for t secs
                p = subprocess.Popen(
                    ['./try.sh', str(t)])
                print("Sleep ", t)
                time.sleep(t)
                # p.terminate()
            lock.acquire()
            # Now load has finished executing
            # Get the old states
            prev_state = [buff_len, load]
            action = self.select_action(load, buff_len)
            rew = self.get_reward(load, buff_len, action)
            buff_len = max(buff_len - 1, 0)
            # Add buffers
            buffer.add(prev_state, action, [0, 0], rew, 0, 0, 0)
            # print("DEPT State, Action Reward",
            #    prev_state, action, rew)
            # if buffer.ptr >= buffer.max_size - 1:
            #    file_count += 1
            #    buffer.save('buffer_' + str(file_count))
            lock.release()
        else:
            count = request.args.get('count')
            #print ("Offloaded Request")
            # Offload request to dummy server
            resp = requests.get('http://172.17.0.3:3333?count=' + count)
            # lock.acquire()
            #print("Main function Action 1 Lock status ", lock.locked())
            lock.acquire()
            # Store the states and rewards and actions in buffer
            prev_state = [buff_len, load]
            action = self.select_action(load, buff_len)
            rew = self.get_reward(load, buff_len, action)
            buffer.add(prev_state, action, [0, 0], rew, 0, 0, 0)
            # print("DEPT State, Action Reward",
            #    prev_state, action, rew)
            # if buffer.ptr >= buffer.max_size - 1:
            #    file_count += 1
            #    buffer.save('buffer_' + str(file_count))
            lock.release()
            #print("Lock released Main function Action 1")
        return [file_count, buffer.ptr]


buff_size = 20
file_count = 0
buff_len = 0
offload = 0
load = 0
run = 0
batch_size = 10000
replay_size = 10000
state_dim = 2
threshold_req = 17
overload_count = 0
offload_count = 0
overload_vec = []
offload_vec = []
lock = th.Lock()
start_time = time.time()
buffer = ReplayBuffer(state_dim, batch_size, replay_size, 'cpu')
req_thres = np.full((21), threshold_req, dtype=float)
api.add_resource(Greeting, '/')  # Route_1
api.add_resource(Notify, '/notify')  # Route_2

if __name__ == '__main__':
    app.run('0.0.0.0', '3333')
