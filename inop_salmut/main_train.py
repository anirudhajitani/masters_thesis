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
import q_learning 
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


"""
def train_salmut_reset(env, policy, steps, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    training_evaluations = []
    episode_num = 0
    avg_reward = 0.0
    done = True
    training_iters = 0
    state = env.reset()
    for _ in range(int(args.eval_freq)):
        action = policy.select_action(state)
        prev_state = state
        state, reward, done, _ = env.step(action)
        avg_reward += reward
        if done:
            training_eval.append(avg_reward)
            avg_reward = 0.0
            np.save(f"./{args.folder}/results/salmut_train_{setting}", training_eval)
        policy.train(prev_state, action, reward, state,
                     args.eval_freq, args.env_name, args.folder)
"""

"""
SALMUT TRAINING 
"""
def train_salmut(env, policy, steps, args, state, j):
    # For saving files
    setting = f"{args.env_name}_{j}"
    training_eval = []
    episode_num = 0
    avg_reward = 0.0
    done = True
    training_iters = 0
    #state = env.reset()
    for _ in range(int(args.eval_freq)):
        action = policy.select_action(state)
        prev_state = state
        state, reward, done, _ = env.step(action)
        avg_reward += reward
        # Current implementation never sets done to False, we can convert this to episodic
        # task by setting this to True once episode ends and computing training rewards too.
        if done:
            training_eval.append(avg_reward)
            avg_reward = 0.0
            np.save(f"./{args.folder}/results/salmut_train_{setting}", training_eval)
        policy.train(prev_state, action, reward, state,
                     args.eval_freq, args.env_name, args.folder, j)
    return state

def train_q_learning(env, policy, steps, args, state, j, i):
    # For saving files
    setting = f"{args.env_name}_{j}"
    training_eval = []
    episode_num = 0
    avg_reward = 0.0
    done = True
    training_iters = 0
    #state = env.reset()
    for _ in range(int(args.eval_freq)):
        action = policy.select_action(state)
        prev_state = state
        state, reward, done, _ = env.step(action)
        avg_reward += reward
        # Current implementation never sets done to False, we can convert this to episodic
        # task by setting this to True once episode ends and computing training rewards too.
        if done:
            training_eval.append(avg_reward)
            avg_reward = 0.0
            np.save(f"./{args.folder}/results/q_learning_train_{setting}", training_eval)
        policy.train(prev_state, action, reward, state,
                     args.eval_freq, args.env_name, args.folder, j, i)
    return state



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# Implemented in main_eval file
"""
def eval_policy(policy, env_name, seed, type=0, eval_episodes=100, threshold_pol=7):
    #eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
    #eval_env.seed(seed + 100)
    global cpu_util
    global action_list
    eval_env = OffloadEnv()
    avg_reward = 0.
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
            avg_reward += reward
            print("Eval policy action reward",
                  prev_state, action, reward, state)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    #cpu_npy = np.array(cpu_util)
    #act_npy = np.array(action_list)
    #np.save('./buffers/cpu_util.npy', cpu_npy)
    #np.save('./buffers/action.npy', act_npy)
    return avg_reward
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    # Algo 0-PPO 1-A2C 2-SAC 3-SALMUT 4-generate_N_lambd
    parser.add_argument("--algo", default=0, type=int)
    parser.add_argument("--baseline-threshold", default=18, type=int)
    parser.add_argument("--env_name", default="res_try_env")
    parser.add_argument("--logdir", default="res_try_log")
    parser.add_argument("--lambd", default=0.5, type=float)
    parser.add_argument("--lambd_high", default=0.75, type=float)
    parser.add_argument("--lambd_evolve", default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--user_identical", default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--user_evolve", default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--folder", default='res_try_0')
    parser.add_argument("--train_iter", default=1e6, type=int)
    parser.add_argument("--eval_freq", default=1e3, type=int)
    parser.add_argument("--offload_cost", default=1.0, type=float)
    parser.add_argument("--overload_cost", default=10.0, type=float)
    parser.add_argument("--holding_cost", default=0.12, type=float)
    parser.add_argument("--reward", default=0.2, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--n_steps", default=1000, type=int)
    parser.add_argument("--N", default=24, type=int)
    args = parser.parse_args()

    # Create directories if does not exists
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    if not os.path.exists(f"./{args.folder}/results"):
        os.makedirs(f"./{args.folder}/results")

    if not os.path.exists(f"./{args.folder}/models"):
        os.makedirs(f"./{args.folder}/models")

    if not os.path.exists(f"./{args.folder}/buffers"):
        os.makedirs(f"./{args.folder}/buffers")

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    print("Lambda Evolve ", args.lambd_evolve, " User Identical ",
          args.user_identical, " User evolve ", args.user_evolve)

    # Intialize state and action dimensions

    state_dim = 2
    num_actions = 2
    #eval_env = OffloadEnv(True, args.lambd, args.mdp_evolve, args.user_evolve, args.user_identical, args.env_name)

    env_name = args.env_name
    setting = f"{env_name}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    #env = DummyVecEnv([lambda: env])
    #log_dir = "./off_a2c_res_5/"

    log_dir = args.logdir

    #env = make_vec_env(lambda: env, n_envs=1)
    #eval_env = make_vec_env(lambda: eval_env, n_envs=1)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)

    loop_range = int(args.train_iter / args.eval_freq)
    # 10 different training iterations
    for j in range(0, 10):
        print("RANDOM SEED ", j)
        lambd = []
        N = []
        # Initialize the environment
        env = OffloadEnv(False, args.lambd, args.offload_cost,
                         args.overload_cost, args.holding_cost, args.reward, args.N, j, args)
        env = Monitor(env, log_dir)
        # env.seed(j)
        # torch.manual_seed(j)
        np.random.seed(j)

        """
        When args.algo == 4, it does not run any experiments.
        It generates the lambda and N for different scenarios 
        based on the values of user_evolve, lambda_evolve, and 
        user_identical

        Scenario    lambda_evolve   user_identical  user_evolve
            1           False           True            False
            2           True            True            False
            3           True            False           False
            4           False           True            True
            5           True            True            True
            6           True            False           True
        
        Scenario 1 -> fixed lambda and fix N
        Scenario 2 -> fixed N, lambda is lambda for first and last 1/3rd
                    timesteps and lambda_high for middle 1/3rd timesteps
        Scenario 3 -> Each user starts with lambda or lambda_high and 
                    toggles lambda every 10^4 intervals with p = 0.1
        Scenario 4 -> Same as scenario 1, N changes every 10^5 timesteps.
                    a user may leave the network, stay in the network or 
                    add another mobile device to the network with
                    probabilities 0.05, 0.9, and 0.05, respectively.
        Scenario 5 -> Scenario 2 + changing users
        Scenario 6 -> Scenario 3 + changing users
        
        """

        if args.algo != 4:
            # Load the lambda and N buffers if we need to run RL algorithms
            with open(f"./{args.folder}/buffers/lambda.npy", "rb") as fp:
                lambd = pickle.load(fp)
            with open(f"./{args.folder}/buffers/N.npy", "rb") as fp:
                N = pickle.load(fp)
        # Intialize models if need to run RL algos
        if args.algo == 0:
            model = PPO('MlpPolicy', env, verbose=0,
                        gamma=0.95, tensorboard_log=log_dir)
        elif args.algo == 1:
            model = A2C('MlpPolicy', env, verbose=0, gamma=0.95,
                        learning_rate=args.lr, n_steps=args.n_steps, tensorboard_log=log_dir)
        elif args.algo == 2:
            model = q_learning.q_learning(num_actions, state_dim)
        elif args.algo == 3:
            # Initialize SALMUT
            model = structured_learning.structured_learning(num_actions, state_dim)

        state = env.reset()
        # Run training iteration
        for i in range(loop_range):
            #print ("TRAIN ", i)
            # Generate lambda and N buffers only when algo == 4
            if args.algo == 4:
                # Scenarios where users are changing (every 10^5 iterations)
                if i > 0 and args.user_evolve == True and i % 100 == 0:
                    # Get current N from environment
                    old_N = env.get_N()
                    new_N = old_N
                    # Get current lambda from environment
                    new_lambd = env.get_lambd()
                    for ru in range(old_N):
                        # Will change users with p = 0.1
                        z = np.random.binomial(n=1, p=0.1, size=1)[0]
                        if z == 1:
                            # If user needs to be changed, will randomly add
                            # one extra device or remove users with p = 0.5
                            p = np.random.binomial(n=1, p=0.5, size=1)[0]
                            if p == 1:
                                new_N += 1
                                new_lambd.append(args.lambd)
                            else:
                                new_N -= 1
                                del new_lambd[-1]
                    # Setting new values of user
                    env.set_N(new_N, new_lambd)
                    print("USER EVOLVE ", env.get_N(), env.get_lambd())
                # Scenarios where lambda is changing
                if i > 0 and args.lambd_evolve == True and i % 10 == 0:
                    curr_N = env.get_N()
                    # If users are not identical (Scenario 3, 6)
                    if args.user_identical == False:
                        new_lambd = []
                        for x in range(curr_N):
                            # Toggle lambda with p = 0.1
                            p = np.random.binomial(n=1, p=0.1, size=1)[0]
                            if p == 0:
                                new_lambd.append(args.lambd)
                            else:
                                new_lambd.append(args.lambd_high)
                        env.set_lambd(new_lambd)
                    else:
                        # When users are identical and lambda evolve (Sc 2, 5)
                        print(i, loop_range/3, loop_range/3 * 2)
                        if i > (loop_range/3) and i < (loop_range/3 * 2):
                            new_lambd = [args.lambd_high] * curr_N
                        else:
                            new_lambd = [args.lambd] * curr_N
                        env.set_lambd(new_lambd)
                    print("LAMBDA EVOLVE ", env.get_lambd())
                lambd.append(env.get_lambd())
                N.append(env.get_N())

                # Save the lambda and N buffers in disk
                with open(f"./{args.folder}/buffers/lambda.npy", "wb") as fp:
                    pickle.dump(lambd, fp)
                with open(f"./{args.folder}/buffers/N.npy", "wb") as fp:
                    pickle.dump(N, fp)
            else:
                # If not generating lambda and N, load its current value for the step.
                env.set_N(int(N[i]), list(lambd[i]))

            # For RL algos except SALMUT, start training using stable-baselines API
            if args.algo != 3 and args.algo != 4 and args.algo != 2:
                model.learn(total_timesteps=args.eval_freq, log_interval=10,
                            reset_num_timesteps=False)
                # Saving models intermittently
                model_name = f"./{args.folder}/models/model_{args.algo}_{j}_{i}"
                model.save(model_name)

            # Loading the previous stored models for PPO, A2C
            if args.algo == 0:
                model = PPO.load(model_name, env)
            elif args.algo == 1:
                model = A2C.load(model_name, env)
            elif args.algo == 2:
                state = train_q_learning(
                    env, model, args.eval_freq, args, state, j, i)
            elif args.algo == 3:
                # Training SALMUT
                state = train_salmut(
                    env, model, args.eval_freq, args, state, j)

        if args.algo == 4:
            exit(1)
