import argparse
import copy
import importlib
import json
import os
import statistics
import numpy as np
import torch
import random
import structured_learning
from NewOffloadEnv import OffloadEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import utils
import pickle
from stable_baselines3.common.cmd_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, SAC
from stable_baselines3 import A2C, PPO
#from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines import DQN
from stable_baselines3.common import results_plotter

# Earlier implementation
"""
def train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    # buffer_name = f"{args.buffer_name}"
    algo = args.algo
    baseline = False
    # Initialize and load policy
    if algo == 0:
        policy = discrete_BCQ.discrete_BCQ(
            # policy = structured_learning.structured_learning(
            is_atari,
            num_actions,
            state_dim,
            device,
            args.BCQ_threshold,
            parameters["discount"],
            parameters["optimizer"],
            parameters["optimizer_parameters"],
            parameters["polyak_target_update"],
            parameters["target_update_freq"],
            parameters["tau"],
            parameters["initial_eps"],
            parameters["end_eps"],
            parameters["eps_decay_period"],
            parameters["eval_eps"]
        )
    elif algo == 1:
        baseline = True
    elif algo == 2:
        baseline = True

    # Load replay buffer
    replay_buffer.load(f"./buffers/{buffer_name}")
    # replay_buffer.load(f"./results/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0
    while training_iters < args.max_timesteps:
        if baseline == False:
            for _ in range(int(parameters["eval_freq"])):
                policy.train(replay_buffer)
        evaluations.append(eval_policy(
            policy, args.env, args.seed, baseline, 100, args.threshold))
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters}")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, type=0, eval_episodes=100, threshold_pol=7):
    #eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
    #eval_env.seed(seed + 100)
    global cpu_util
    global action_list
    eval_env = OffloadEnv()
    avg_dis_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        for t in range(200):
            if type == 0:
                action = policy.select_action(np.array(state), eval=True)
            elif type == 1:
                if state[1] < threshold_pol:
                    action = 0
                else:
                    action = 1
            prev_state = state
            # cpu_util.append(state[1])
            # action_list.append(action)
            state, reward, done, _ = eval_env.step(action)
            avg_dis_reward += reward
            print("Eval policy action reward",
                  prev_state, action, reward, state)

    avg_dis_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_dis_reward:.3f}")
    print("---------------------------------------")
    #cpu_npy = np.array(cpu_util)
    #act_npy = np.array(action_list)
    #np.save('./buffers/cpu_util.npy', cpu_npy)
    #np.save('./buffers/action.npy', act_npy)
    return avg_dis_reward

"""

if __name__ == "__main__":

    # Load parameters
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="AdmissionControl")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=20, type=int)
    # Prepends name to filename
    parser.add_argument("--buffer_name", default="offload_0310")
    # Max time steps to run environment or train for
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Algo 0-PPO 1-A2C 2-SAC
    parser.add_argument("--algo", default=0, type=int)
    parser.add_argument("--baseline-threshold", default=18, type=int)
    parser.add_argument("--env_name", default="res_try_env")
    parser.add_argument("--logdir", default="res_try_log")
    parser.add_argument("--lambd", default=0.5, type=float)
    parser.add_argument("--lambd_high", default=0.75, type=float)
    parser.add_argument("--lambd_evolve", default=False, type=bool)
    parser.add_argument("--user_evolve", default=False, type=bool)
    parser.add_argument("--user_identical", default=True, type=bool)
    parser.add_argument("--folder", default='res_try_0')
    parser.add_argument("--train_iter", default=1e6, type=int)
    parser.add_argument("--eval_freq", default=1e3, type=int)
    parser.add_argument("--offload_cost", default=1.0, type=float)
    parser.add_argument("--overload_cost", default=10.0, type=float)
    parser.add_argument("--holding_cost", default=0.12, type=float)
    parser.add_argument("--reward", default=0.2, type=float)
    parser.add_argument("--N", default=24, type=int)
    # No of steps of evaluations to run
    parser.add_argument("--step", default=1000, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    args = parser.parse_args()
    args = parser.parse_args()

    print("---------------------------------------")

    # Already created these files during training
    """
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    
    if not os.path.exists(f"./{args.folder}/results"):
        os.makedirs(f"./{args.folder}/results")

    if not os.path.exists("./{args.folder}/models"):
        os.makedirs(f"./{args.folder}/models")

    if not os.path.exists("./{args.folder}/buffers"):
        os.makedirs(f"./{args.folder}/buffers")

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    """

    # Define state dimensions, action dimensions and discount factor
    state_dim = 2
    num_actions = 2
    gamma = 0.95

    # Create new evaluation environment
    env = OffloadEnv(True, args.lambd, args.offload_cost, args.overload_cost, args.holding_cost,
                     args.reward, args.N, args.seed, args.env_name, args.folder, args.start_iter, args.step)
    #eval_env = OffloadEnv(True, args.lambd, args.mdp_evolve, args.user_evolve, args.user_identical, args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SALMUT Algorithm class intialization
    if args.algo == 3:
        model = structured_learning.structured_learning(num_actions, state_dim)
    #env_name = "offload_dqn_mdp_5"

    # Initializing variables
    env_name = args.env_name
    setting = f"{env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Storing medians
    testing_eval_med = []
    #eval_env = make_vec_env(lambda: eval_env, n_envs=1)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)

    loop_range = int(args.train_iter / args.eval_freq)

    # Loading lambda and N for simulations

    with open(f"./{args.folder}/buffers/lambda.npy", "rb") as fp:
        lambd = pickle.load(fp)
    with open(f"./{args.folder}/buffers/N.npy", "rb") as fp:
        N = pickle.load(fp)

    """
    We can run evaluations in parallel after training has completed.
    We need to specify the start loop and end loop.
    It will create separate files in the results folder.
    Need to integrate the files manually if we do it this way.
    """

    # Where to start evaluation iteration
    start_loop = args.start_iter
    # Where to end evaluations
    end_loop = args.start_iter + args.step
    print("START , END", start_loop, end_loop)
    for i in range(start_loop, end_loop):
        print("EVAL ", i)
        avg_dis_reward_run = []
        for j in range(0, 10):
            print("SEED 0")
            # Loading models for RL algorithms
            model_name = f"./{args.folder}/models/model_{args.algo}_{j}_{i}"
            #print ("Lambd N i ", lambd[i], N[i])

            # Setting the values of N and lambda in the environment for this step
            env.set_N(int(N[i]), list(lambd[i]))

            # Loading the saved training models for evaluation
            if args.algo == 0:
                model = PPO.load(model_name, env)
            elif args.algo == 1:
                model = A2C.load(model_name, env)
            elif args.algo == 2:
                model = SAC.load(model_name, env)
            elif args.algo == 3:
                # For SALMUT we load the threshold vector policies
                thres_vec = np.load(f"./{args.folder}/buffers/thresvec_{args.env_name}_{j}.npy")
                # Setting the threshold vectors for SALMUT
                model.set_threshold_vec(thres_vec[i])
            avg_dis_reward = 0.0

            # Run each eval step for 100 different seeds
            for k in range(100):
                env.seed(k)
                obs = env.reset()
                reward_traj = []
                dis_reward = 0.0
                # We fix each step to 1000 iterations
                for t in range(int(1e3)):
                    if args.algo == 3:
                        # For SALMUT this gives us the action
                        action = model.select_action(np.array(obs), eval_=True)
                    else:
                        # For other RL algorithms
                        action, _states = model.predict(obs)
                    # Take a step in the environment with the action
                    obs, reward, done, info = env.step(action)
                    # Add to reward trajectory
                    reward_traj.append(reward)
                    #print(obs, reward, done, info)

                # Compute the discounted reward for this evaluation
                for r in reward_traj[::-1]:
                    dis_reward = r + gamma * dis_reward
                # Adding curr dis_reward avg_dis_reward which will compute the average
                avg_dis_reward += dis_reward
                #print ("Reward", avg_dis_reward)

            # Performing average of 100 runs
            avg_dis_reward = avg_dis_reward / 100
            print("AVG REWARD", avg_dis_reward)
            # Adding rewards to list
            avg_dis_reward_run.append(avg_dis_reward)
        # Computing the quartiles
        quartiles = np.percentile(avg_dis_reward_run, [25, 50, 75])
        print("quartiles ", quartiles)
        testing_eval_med.append(quartiles)
        # testing_eval_std.append(avg_std)
        # Saving the results
        np.save(f"./{args.folder}/results/median_{start_loop}_{end_loop}_{setting}.npy", testing_eval_med)
