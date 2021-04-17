import numpy as np
import sys

folder = sys.argv[1]
env_name = sys.argv[2]

rew_run = []
rewards = []
ov_run = []
ov = []
off_run = []
off = []
gamma = 0.99

for i in range(1, 1000):
    rew_run = []
    ov_run = []
    off_run = []
    for j in range(1, 6):
        ov_buff = np.load(f'./{folder}/buffers/buffer_{j}_overload_count.npy')
        off_buff = np.load(f'./{folder}/buffers/buffer_{j}_offload_count.npy')
        rew_buff = np.load(f'./{folder}/buffers/buffer_{j}_{i}_reward.npy')
        reward_traj = list(rew_buff)
        dis_reward = 0.0
        for r in reward_traj[::-1]:
            dis_reward = r + gamma * dis_reward
        #print("Dis reward", dis_reward, type(dis_reward))
        rew_run.append(float(dis_reward))
        ov_run.append(ov_buff[i-1])
        off_run.append(off_buff[i-1])
        print(i, j, ov_buff, off_buff)
        print(i, j, ov_run, off_run)
    rewards.append(np.percentile(rew_run, [25, 50, 75]))
    ov.append(np.percentile(ov_run, [25, 50, 75]))
    off.append(np.percentile(off_run, [25, 50, 75]))

np.save(f'./{folder}/results/rewards_train_{env_name}.npy', rewards)
np.save(f'./{folder}/results/overload_train_{env_name}.npy', ov)
np.save(f'./{folder}/results/offload_train_{env_name}.npy', off)
