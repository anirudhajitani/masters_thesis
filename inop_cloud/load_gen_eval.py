import random
import time
import subprocess
from random import randrange
import sys
import re
from multiprocessing import Process
import threading as th
import numpy as np
import pickle

ip_address = '172.17.0.2'
port = '3333'
container_name = 'app1'
folder = sys.argv[1]
env_name = sys.argv[2]
fc = 0
res_path = f'./{folder}/results/rewards_{env_name}.npy'
ov_path = f'./{folder}/results/overload_{env_name}.npy'
off_path = f'./{folder}/results/offload_{env_name}.npy'
results_run = []
results = []
ov_run = []
ov = []
off_run = []
off = []


# If results already exists, will load them
# Ensure to delete previous results or rename them before starting a new run
try:
    off = list(np.load(off_path))
    ov = list(np.load(ov_path))
    results = list(np.load(res_path))
    start_loop = len(results)
except:
    off = []
    ov = []
    results = []
    start_loop = 0


def fireEvent(start_time):
    global fc
    x = randrange(0, 1)
    print(x, time.time() - start_time)
    q_str = 'http://' + ip_address + ':' + port + '?' + 'count=' + str(x)
    out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-s', q_str],
                           # out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-w', '@curlformat', '-s', q_str],
                           # out = subprocess.Popen(['docker', 'run', '--rm', 'curl_client', '-w', '@curlformat', '-o', '/dev/null', '-s', q_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    print(stdout)
    print(stderr)
    # Get new FC from stdout


def run_rl_module_and_notify(fc, run, eval_run):
    global results_run
    global results
    global ov_run
    global ov
    global start_loop
    global off_run
    global off
    dest_path_str = container_name + ':/req_thres.npy'
    src_path_str = f'./{folder}/buffers/thresvec_{str(run)}_{env_name}_{str(fc)}.npy'
    out = subprocess.Popen(['docker', 'cp', src_path_str, dest_path_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    print(stdout)
    print(stderr)
    q_str = 'http://' + ip_address + ':' + port + \
        '/notify?' + 'offload=' + str(eval_run)
    # out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-w', '@curlformat', '-s', q_str],
    out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-s', q_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    #if run == 1 and eval_run == 1:
        # Uncomment this, done this just for once
        #if fc == start_loop:
        #    return
    if fc == 0:
        return

    q_str = 'http://' + ip_address + ':' + port + '/notify?' + 'offload=0'
    # out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-w', '@curlformat', '-s', q_str],
    out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-s', q_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode('utf-8')
    #op = re.split('\n|\"', stdout)
    op = re.split('\[|\]|, ', stdout)
    print(stdout)
    print(stderr)
    print("Ouput ", op, " OP ", op[1])
    results_run.append(float(op[1]))
    ov_run.append(int(op[2]))
    off_run.append(int(op[3]))
    # Perform median calculation (need to do 1 loop later)
    # if run == 5 and eval_run == 5:
    # if run == 1 and eval_run == 1:
    if eval_run == 1:
        #avg_dis_rew = np.percentile(results_run, [25,50,75])
        #avg_ov = np.percentile(ov_run, [25,50,75])
        #avg_off = np.percentile(off_run, [25,50,75])
        print("AVG DIS REWARD : ", results_run)
        print("OV OC : ", ov_run, off_run)
        results.append(results_run)
        ov.append(ov_run)
        off.append(off_run)
        np.save(res_path, results)
        np.save(ov_path, ov)
        np.save(off_path, off)
        results_run = []
        ov_run = []
        off_run = []


def process_event(lambd):
    start_time = time.time()
    while time.time() - start_time < 100:
        #interval = random.expovariate(0.1)
        interval = random.expovariate(lambd)
        interval = min(interval, 20.0)
        print("Interval ", interval, lambd)
        time.sleep(interval)
        fireEvent(start_time)


def main():
    fc = 0
    global start_loop
    with open(f"./{folder}/buffers/lambda.npy", "rb") as fp:
        lambd = pickle.load(fp)
    with open(f"./{folder}/buffers/N.npy", "rb") as fp:
        N = pickle.load(fp)
    #start_loop = 0
    for l in range(start_loop, 1000):
        for run in range(1, 6):
            for eval_run in range(1, 6):
                random.seed(eval_run)
                print("STEP ", l, " TRAIN_RUN ", run, " EVAL_RUN ", eval_run)
                run_rl_module_and_notify(l, run, eval_run)
                jobs = []
                for i in range(N[l]):
                    x = lambd[l][i] / 2.0
                    print(x)
                    t = th.Thread(target=process_event, args=(x,))
                    jobs.append(t)
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join(timeout=40)


if __name__ == "__main__":
    main()
