from matplotlib import pyplot as plt
import numpy as np
import sys
import os
from scipy.signal import savgol_filter
from collections import OrderedDict

if len(sys.argv) < 2:
    print ("Provide folder name")
    exit(1)
folder = sys.argv[1]
os.chdir(f"./{folder}/results")

"""
# Loading median results
y1 = np.load('median_ppo_eval_20.npy')
y2 = np.load('median_a2c_eval_20.npy')
y3 = np.load('median_salmut_20.npy')
y4 = np.load('median_plan_eval_20.npy')
y5 = np.load('median_thres_eval_20.npy')
"""

a2c_ov = []
a2c_off = []
ppo_ov = []
ppo_off = []
salmut_ov = []
salmut_off = []
q_learning_ov = []
q_learning_off = []

for i in range(10):
    a2c_ov.append(np.load(f'overload_med_a2c_{i}.npy'))
    a2c_off.append(np.load(f'offload_med_a2c_{i}.npy'))
    ppo_ov.append(np.load(f'overload_med_ppo_{i}.npy'))
    ppo_off.append(np.load(f'offload_med_ppo_{i}.npy'))
    salmut_ov.append(np.load(f'overload_med_salmut_{i}.npy'))
    salmut_off.append(np.load(f'offload_med_salmut_{i}.npy'))
    q_learning_ov.append(np.load(f'overload_med_q_learning_{i}.npy'))
    q_learning_off.append(np.load(f'offload_med_q_learning_{i}.npy'))

print (len(a2c_ov), len(a2c_off), type(a2c_ov), type(a2c_off))
a2c_ov_med = np.array(np.percentile(a2c_ov, [25, 50, 75], axis=0))
print (a2c_ov_med, type(a2c_ov_med), a2c_ov_med.shape)
a2c_off_med = np.array(np.percentile(a2c_off, [25, 50, 75], axis=0))
ppo_ov_med = np.array(np.percentile(ppo_ov, [25, 50, 75], axis=0))
ppo_off_med = np.array(np.percentile(ppo_off, [25, 50, 75], axis=0))
salmut_ov_med = np.array(np.percentile(salmut_ov, [25, 50, 75], axis=0))
salmut_off_med = np.array(np.percentile(salmut_off, [25, 50, 75], axis=0))
q_learning_ov_med = np.array(np.percentile(q_learning_ov, [25, 50, 75], axis=0))
q_learning_off_med = np.array(np.percentile(q_learning_off, [25, 50, 75], axis=0))
plan_ov = np.load('overload_med_plan_eval_20.npy')
plan_off = np.load('offload_med_plan_eval_20.npy')
thres_ov = np.load('overload_med_thres_eval_20.npy')
thres_off = np.load('offload_med_thres_eval_20.npy')

np.save('overload_med_a2c_combined.npy', a2c_ov_med)
np.save('offload_med_a2c_combined.npy', a2c_off_med)
np.save('overload_med_ppo_combined.npy', ppo_ov_med)
np.save('offload_med_ppo_combined.npy', ppo_off_med)
np.save('overload_med_salmut_combined.npy', salmut_ov_med)
np.save('offload_med_salmut_combined.npy', salmut_off_med)
np.save('overload_med_q_learning_combined.npy', salmut_ov_med)
np.save('offload_med_q_learning_combined.npy', salmut_off_med)

x = [i*1000 for i in range(0, 1000)]

a2c_ov_med = savgol_filter(a2c_ov_med, 21, 4) # window size 51, polynomial order 3
a2c_off_med = savgol_filter(a2c_off_med, 21, 4) # window size 51, polynomial order 3
ppo_ov_med = savgol_filter(ppo_ov_med, 21, 4) # window size 51, polynomial order 3
ppo_off_med = savgol_filter(ppo_off_med, 21, 4) # window size 51, polynomial order 3
salmut_ov_med = savgol_filter(salmut_ov_med, 21, 4) # window size 51, polynomial order 3
salmut_off_med = savgol_filter(salmut_off_med, 21, 4) # window size 51, polynomial order 3
q_learning_ov_med = savgol_filter(q_learning_ov_med, 21, 4) # window size 51, polynomial order 3
q_learning_off_med = savgol_filter(q_learning_off_med, 21, 4) # window size 51, polynomial order 3
plan_ov = savgol_filter(plan_ov, 21, 4) # window size 51, polynomial order 3
plan_off = savgol_filter(plan_off, 21, 4) # window size 51, polynomial order 3
thres_ov = savgol_filter(thres_ov, 21, 4) # window size 51, polynomial order 3
thres_off = savgol_filter(thres_off, 21, 4) # window size 51, polynomial order 3


fig, ax = plt.subplots(figsize=(7,4))
ax.plot(x, thres_ov, '#f781bf', label='Baseline')
ax.plot(x, a2c_ov_med[1,:], '#ff7f00', label='A2C')
ax.fill_between(x, a2c_ov_med[0,:], a2c_ov_med[2,:], color='#ff7f00', alpha=0.2)
ax.plot(x, ppo_ov_med[1,:], '#4daf4a', label='PPO')
ax.fill_between(x, ppo_ov_med[0,:], ppo_ov_med[2,:], color='#4daf4a', alpha=0.2)
ax.plot(x, q_learning_ov_med[1,:], '#653700', label='QL')
ax.fill_between(x, q_learning_ov_med[0,:], q_learning_ov_med[2,:], color='#653700', alpha=0.2)
ax.plot(x, plan_ov, '#e41a1c', label='DP')
ax.plot(x, salmut_ov_med[1,:], '#377eb8', label='SALMUT')
ax.fill_between(x, salmut_ov_med[0,:], salmut_ov_med[2,:], color='#377eb8', alpha=0.2)
##ax.plot(x, -y6[1,:], 'c-', label='MPC')
##ax.fill_between(x, -y6[2,:], -y6[0,:], color='c', alpha=0.2)
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
labels = list(by_label.values())[::-1] #reverse label order
handles = list(by_label.keys())[::-1] #reverse handle order
labels[0], labels[1] = labels[1], labels[0]
handles[0], handles[1] = handles[1], handles[0]
ax.set_xlabel('Timesteps')
ax.set_ylabel('Overloaded States')
ax.set_ylim(ymin=0, ymax=100)
#plt.legend(loc='best')
plt.legend(labels, handles, bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center',
                   ncol=6, mode="expand", borderaxespad=0.)
fig.savefig(f"{folder}_overload.png")

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(x, thres_off, '#f781bf', label='Baseline')
ax.plot(x, a2c_off_med[1,:], '#ff7f00', label='A2C')
ax.fill_between(x, a2c_off_med[0,:], a2c_off_med[2,:], color='#ff7f00', alpha=0.2)
ax.plot(x, ppo_off_med[1,:], '#4daf4a', label='PPO')
ax.fill_between(x, ppo_off_med[0,:], ppo_off_med[2,:], color='#4daf4a', alpha=0.2)
ax.plot(x, q_learning_off_med[1,:], '#653700', label='QL')
ax.fill_between(x, q_learning_off_med[0,:], q_learning_off_med[2,:], color='#653700', alpha=0.2)
ax.plot(x, plan_off, '#e41a1c', label='DP')
ax.plot(x, salmut_off_med[1,:], '#377eb8', label='SALMUT')
ax.fill_between(x, salmut_off_med[0,:], salmut_off_med[2,:], color='#377eb8', alpha=0.2)
##ax.plot(x, -y6[1,:], 'c-', label='MPC')
##ax.fill_between(x, -y6[2,:], -y6[0,:], color='c', alpha=0.2)
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
labels = list(by_label.values())[::-1] #reverse label order
handles = list(by_label.keys())[::-1] #reverse handle order
labels[0], labels[1] = labels[1], labels[0]
handles[0], handles[1] = handles[1], handles[0]
ax.set_xlabel('Timesteps')
ax.set_ylabel('Offloading Count')
ax.set_ylim(ymin=0, ymax=400)
#plt.legend(loc='best')
plt.legend(labels, handles, bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center',
                   ncol=5, mode="expand", borderaxespad=0.)
fig.savefig(f"{folder}_offload_new.png")

"""

fig, ax = plt.subplots()
ax.plot(x, ov1[:,1], 'r-', label='PPO')
ax.fill_between(x, ov1[:,0], ov1[:,2], color='r', alpha=0.2)
ax.plot(x, ov2[:,1], 'g-', label='A2C')
ax.fill_between(x, ov2[:,0], ov2[:,2], color='g', alpha=0.2)
ax.plot(x, ov3[:,1], 'b-', label='SALMUT')
ax.fill_between(x, ov3[:,0], ov3[:,2], color='b', alpha=0.2)
ax.plot(x, ov4[:,1], 'c-', label='DP')
ax.fill_between(x, ov4[:,0], ov4[:,2], color='c', alpha=0.2)
ax.plot(x, ov5[:,1], 'm-', label='Threshold=18')
ax.fill_between(x, ov5[:,0], ov5[:,2], color='m', alpha=0.2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center',
                           ncol=5, mode="expand", borderaxespad=0.)
plt.xlabel('Timesteps')
plt.ylabel('Overloaded State Count') 
fig.savefig(f"{folder}_overload.png")


fig, ax = plt.subplots()
ax.plot(x, off1[:,1], 'r-', label='PPO')
ax.fill_between(x, off1[:,0], off1[:,2], color='r', alpha=0.2)
ax.plot(x, off2[:,1], 'g-', label='A2C')
ax.fill_between(x, off2[:,0], off2[:,2], color='g', alpha=0.2)
ax.plot(x, off3[:,1], 'b-', label='SALMUT')
ax.fill_between(x, off3[:,0], off3[:,2], color='b', alpha=0.2)
ax.plot(x, off4[:,1], 'c-', label='DP')
ax.fill_between(x, off4[:,0], off4[:,2], color='c', alpha=0.2)
ax.plot(x, off5[:,1], 'm-', label='Threshold=18')
ax.fill_between(x, off5[:,0], off5[:,2], color='m', alpha=0.2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='upper center',
                           ncol=5, mode="expand", borderaxespad=0.)
if 'scenario_1' in folder or 'scenario_3' in folder or 'scenario_4' in folder:
    plt.ylim(0,200)
plt.xlabel('Timesteps')
plt.ylabel('Requests Offloaded')
fig.savefig(f"{folder}_offload.png")
"""
