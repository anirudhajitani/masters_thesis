import random
import time
import subprocess
from random import randrange
import sys
import re
from multiprocessing import Process
import threading as th
import pickle

# Set IP address, port, container name
ip_address = '172.17.0.2'
port = '3333'
container_name = 'app1'

# Folder and algorithm respectively
folder = sys.argv[1]
env_name = sys.argv[2]
fc = 0


def fireEvent(start_time):
    """
    Creates a new docker image which send curl request to edge node to execute
    """
    global fc
    x = randrange(0, 1)
    #print (x, time.time() - start_time)
    q_str = 'http://' + ip_address + ':' + port + '?' + 'count=' + str(x)
    # out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-w', '@curlformat', '-s', q_str],
    # out = subprocess.Popen(['docker', 'run', '--rm', 'curl_client', '-w', '@curlformat', '-o', '/dev/null', '-s', q_str],
    # Create a new request by sending CURL-request
    out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-s', q_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    stdout = stdout.decode('utf-8')
    op = re.split('\[|, ', stdout)
    print(op)
    print(stdout)
    print(stderr)
    new_fc = int(op[1])
    # if we want to run it based on buffer overflow and not time-based, we use this
    # in this version, it will be called later (new_fc will always be equal to fc)
    # need to change implementation in app1 by changing file count once full
    # We choose a very high buffer size in app2.py so that this condition doesn't happen
    """
    if new_fc > fc:
        fc = new_fc
        run_rl_module_and_notify(fc)
    """

def run_rl_module_and_notify(fc, run):
    """
    One step of the run finished. Then we need to send notification to edge node.
    """
    print("Notification module called")

    # Send the run id for random seed purpose in application
    q_str = 'http://' + ip_address + ':' + \
        port + '/notify?' + 'offload=' + str(-run)
    # out = subprocess.Popen(['docker', 'run', '--rm', 'curl_client', '-w', '@curlformat', '-s', q_str],
    out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-s', q_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    print(stdout)
    print(stderr)
    # If first step, then we exit here
    if fc == 0:
        return

    # Offload with fc (current step) tells edge node to save its buffers and push it to disk
    q_str = 'http://' + ip_address + ':' + \
        port + '/notify?' + 'offload=' + str(fc)
    # out = subprocess.Popen(['docker', 'run', '--rm', 'curl_client', '-w', '@curlformat', '-s', q_str],
    out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-s', q_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    print(stdout)
    print(stderr)

    # We need to copy the buffers from edge node to VM
    files_dst = ['_ptr.npy', '_state.npy', '_next_state.npy',
                 '_action.npy', '_reward.npy', '_not_done.npy']
    files_dst_2 = ['overload_count.npy', 'offload_count.npy']
    dest_path_str = folder + '/buffers/'
    for file_dst in files_dst:
        path_str = container_name + ':/buffer_' + \
            str(run) + '_' + str(fc) + file_dst
        out = subprocess.Popen(['docker', 'cp', path_str, dest_path_str],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        print(stdout)
        print(stderr)
    for file_dst in files_dst_2:
        path_str = container_name + ':/buffer_' + str(run) + '_' + file_dst
        out = subprocess.Popen(['docker', 'cp', path_str, dest_path_str],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        print(stdout)
        print(stderr)

    # Run SALMUT algorithm, calls main_train which will invoke SALMUT
    buffer_name = 'buffer_' + str(run) + '_' + str(fc)
    out = subprocess.Popen(['python3.6', 'main_train.py', '--replay_buffer', buffer_name, '--fc', str(fc),
                            '--run', str(run), '--algo', '3', '--folder', folder, '--env_name', env_name],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    # print(stdout)
    # print(stderr)

    # Copies updated threshold to the edge node
    dest_path_str = container_name + ':/req_thres.npy'
    src_path_str = f'./{folder}/buffers/thresvec_{str(run)}_{env_name}_{str(fc)}.npy'
    out = subprocess.Popen(['docker', 'cp', src_path_str, dest_path_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    print(stdout)
    print(stderr)

    # Notify edge node of new policy to load (offload value is 0)
    q_str = 'http://' + ip_address + ':' + port + '/notify?' + 'offload=0'
    # out = subprocess.Popen(['docker', 'run', '--rm', 'curl_client', '-w', '@curlformat', '-s', q_str],
    out = subprocess.Popen(['docker', 'run', '--rm', 'byrnedo/alpine-curl', '-s', q_str],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    print(stdout)
    print(stderr)
    print("NOTIF completed")
    return


def process_event(lambd):
    start_time = time.time()
    # RUn each step for 100 secs
    while time.time() - start_time < 100:
        #interval = random.expovariate(0.1)
        """
        if lambd == 0.5:
            lambd = 0.25
        elif lambd == 0.75:
            lambd = 0.375
        """
        # Generate random variable
        interval = random.expovariate(lambd)
        # Max time for a job clip to 20s
        interval = min(interval, 20.0)
        print("Interval = ", interval, lambd)
        # Sleep for intervals and then call fire event
        time.sleep(interval)
        fireEvent(start_time)


def main():
    fc = 0
    """
    Load lambda and N parameters (same way as Gym)
    """
    with open(f"./{folder}/buffers/lambda.npy", "rb") as fp:
        lambd = pickle.load(fp)
    with open(f"./{folder}/buffers/N.npy", "rb") as fp:
        N = pickle.load(fp)
    # Five different iterations
    for run in range(1, 5):
        random.seed(run)
        # Set default start_loop as 0 (can specify start loop)
        start_loop = 0
        for l in range(start_loop, 1000):
            print("RUN = ", run, " LOOP = ", l)
            jobs = []
            # Call first so that random seeds can be initialized and for
            # subsequent calls performs all the necessary steps
            run_rl_module_and_notify(l, run)
            for i in range(N[l]):
                # We use by 2.0 as it gives us a good balance
                x = lambd[l][i] / 2.0
                # Create a new thread and call process event with lambda as parameter
                t = th.Thread(target=process_event, args=(x,))
                # Add thread to jobs
                jobs.append(t)
            for j in jobs:
                j.start()
            for j in jobs:
                # Specify max time limit for run
                j.join(timeout=20)
            print("LOOP ", l, " complete ")


if __name__ == "__main__":
    main()
