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
from plan_policy_new_debug import PlanPolicy
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


def encode_state(obs):
    #print (obs)
    """
    Encoding the state tuple into an integer
    Total of 21 values for load and buffer.
    """
    s = int(obs[1] * 21 + obs[0])
    #print (obs, s)
    return s


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
    # Algo 0-DP 1-Baseline 2-Parameter Estimation
    parser.add_argument("--algo", default=0, type=int)
    parser.add_argument("--baseline-threshold", default=18, type=int)
    parser.add_argument("--env_name", default="plan_res_try_env")
    parser.add_argument("--logdir", default="plan_res_try_log")
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
    args = parser.parse_args()

    # Already done in main_train
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
    state_dim = 2
    num_actions = 2
    gamma = 0.95

    # Defining environment
    env = OffloadEnv(True, args.lambd, args.offload_cost, args.overload_cost,
                     args.holding_cost, args.reward, args.N, args.seed, args.env_name, args.folder)
    #eval_env = OffloadEnv(True, args.lambd, args.mdp_evolve, args.user_evolve, args.user_identical, args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #env_name = "offload_dqn_mdp_5"

    # Setting variables
    env_name = args.env_name
    setting = f"{env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    #env = DummyVecEnv([lambda: env])
    #env = make_vec_env(lambda: env, n_envs=1)
    testing_eval_med = []
    #eval_env = make_vec_env(lambda: eval_env, n_envs=1)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)

    loop_range = int(args.train_iter / args.eval_freq)
    # Loading lambda and N values
    with open(f"./{args.folder}/buffers/lambda.npy", "rb") as fp:
        lambd = pickle.load(fp)
    with open(f"./{args.folder}/buffers/N.npy", "rb") as fp:
        N = pickle.load(fp)

    for i in range(loop_range):
        avg_dis_reward_run = []
        # Setting lambda and N values
        env.set_N(int(N[i]), list(lambd[i]))

        # Algo == 0 is for DP
        if args.algo == 0:
            # Check if policy already saved in disk, if so load the policy
            if os.path.isfile(f"./policies_analysis/policy_plan_lambd_{sum(lambd[i])}.npy"):
                policy = np.load(f"./policies_analysis/policy_plan_lambd_{sum(lambd[i])}.npy")
            else:
                # Initialize PlanPolicy and compute the policy
                pol = PlanPolicy(N[i], lambd[i], args.overload_cost,
                                 args.offload_cost, args.holding_cost, args.reward, gamma=gamma)
                policy = pol.compute_policy(save=True)
                print("Computing Policy")
            policy = list(policy)
        avg_dis_reward = 0.0

        # Run 100 different seeds for each eval step
        for k in range(100):
            env.seed(k)
            obs = env.reset()
            # Parameter Estimation
            if args.algo == 2:
                # Get estimates of resources and lambda from environment
                resources_prob = env.get_prob_resources()
                est_lam = round(env.get_lambda_estimates(), 1)
                print("LAMBDA ESTIMATES", est_lam)
                print("PROB RESOURCES", resources_prob)
                # If optimal policy exists for the given lambda (load it)
                if os.path.isfile(f"./policies/policy_plan_lambd_{est_lam}.npy"):
                    policy = np.load(f"./policies/policy_plan_lambd_{est_lam}.npy")
                    #print (policy, policy.shape)
                else:
                    print("Computing New policy")
                    est_lam_vec = [est_lam/N[i]] * N[i]
                    pol = PlanPolicy(N[i], est_lam_vec, args.overload_cost, args.offload_cost,
                                     args.holding_cost, args.reward, gamma=gamma, prob=resources_prob)
                    policy = pol.compute_policy(save=True)
                policy = list(policy)
            reward_traj = []
            dis_reward = 0.0
            # Run for 1000 steps
            for t in range(int(1e3)):
                # If DP or parameter estimation
                if args.algo == 0 or args.algo == 2:
                    # Encode state
                    state = encode_state(obs)
                    # Get action from policy
                    action = policy[state]
                elif args.algo == 1:
                    # Baseline policy
                    if obs[1] >= args.baseline_threshold:
                        action = 1
                    else:
                        action = 0
                # Take step in the environment
                obs, reward, done, info = env.step(action)
                #print(obs, reward, done, info)
                # Add rewards to reward trajectory
                reward_traj.append(reward)

            # Compute discounted rewards
            for r in reward_traj[::-1]:
                dis_reward = r + gamma * dis_reward
            avg_dis_reward += dis_reward
            #print ("Reward", avg_dis_reward)

        # Get the average of the rewards for 100 runs
        avg_dis_reward = avg_dis_reward / 100
        # Get the quartiles
        quartiles = [avg_dis_reward, avg_dis_reward, avg_dis_reward]
        print("quartiles ", quartiles)
        # Add results to list
        testing_eval_med.append(quartiles)
        # testing_eval_std.append(avg_std)

        # Save result to disk
        np.save(f"./{args.folder}/results/median_{setting}", testing_eval_med)
        # np.save(f"./{args.folder}/results/std_{setting}", testing_eval_std)
