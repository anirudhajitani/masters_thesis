from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import pickle
from scipy.signal import savgol_filter
from collections import OrderedDict

if len(sys.argv) < 2:
    print ("Provide folder name")
    exit(1)
folder = sys.argv[1]

#os.chdir(f"./{folder}/buffers")
with open(f'../inop_cloud/{folder}/buffers/lambda.npy', 'rb') as f:
    L = pickle.load(f)

lambd = []

for i in range(999):
    lambd.append(sum(L[i])/2.0)

os.chdir(f"./{folder}/results")
x = []
for i in range(999):
    x.append(i)

x = np.array(x)
y2 = np.load('rewards_train_salmut.npy')
y1 = np.load('rewards_baseline.npy')
#print(x.shape, y1.shape)
b1 = savgol_filter(y1[0:999, 0], 21, 4)
b2 = savgol_filter(y1[0:999, 1], 21, 4)
b3 = savgol_filter(y1[0:999, 2], 21, 4)
s1 = savgol_filter(y2[0:999, 0], 21, 4)
s2 = savgol_filter(y2[0:999, 1], 21, 4)
s3 = savgol_filter(y2[0:999, 2], 21, 4)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(x, -s2, '#377eb8', label='SALMUT')
ax.fill_between(x, -s3, -s1, color='#377eb8', alpha=0.2)
ax.plot(x, -b2, '#f781bf', label='Baseline')
ax.fill_between(x, -b3, -b1, color='#f781bf', alpha=0.2)
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Timesteps', fontsize=14)
ax.set_ylabel('Discounted Total Cost', fontsize=14)
# ax.legend(labels, handles, bbox_to_anchor=(0., 1.02, 1., .102), prop={'size': 12}, loc='upper center',
#                   ncol=2, mode="expand", borderaxespad=0.)
ax.set_ylim([200, 1200])
ax.legend(loc=1)

ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(x, lambd, color="gray", linestyle="dotted")
#ax2.plot(x, lambd, color="",marker="o")
ax2.set_ylabel(r"$\sum \lambda_i$", color="gray")
#plt.show()


fig.savefig(f"{folder}_dis_reward_testbed.png")

y2 = np.load('overload_train_salmut.npy')
y1 = np.load('overload_baseline.npy')
#y1 = savgol_filter(y1[0:999], 11, 3)
b1 = savgol_filter(y1[0:999, 0], 21, 4)
b2 = savgol_filter(y1[0:999, 1], 21, 4)
b3 = savgol_filter(y1[0:999, 2], 21, 4)
s1 = savgol_filter(y2[0:999, 0], 21, 4)
s2 = savgol_filter(y2[0:999, 1], 21, 4)
s3 = savgol_filter(y2[0:999, 2], 21, 4)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(x, s2, '#377eb8', label='SALMUT')
ax.fill_between(x, s1, s3, color='#377eb8', alpha=0.2)
ax.plot(x, b2, '#f781bf', label='Baseline')
ax.fill_between(x, b1, b3, color='#f781bf', alpha=0.2)
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Timesteps', fontsize=14)
ax.set_ylabel('Overloaded States', fontsize=14)
ax.set_ylim([0, 30])
ax.legend(loc=1)
# ax.legend(labels, handles, bbox_to_anchor=(0., 1.02, 1., .102), prop={'size': 12}, loc='upper center',
#                   ncol=2, mode="expand", borderaxespad=0.)
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(x, lambd, color="gray", linestyle="dotted")
#ax2.plot(x, lambd, color="",marker="o")
ax2.set_ylabel(r"$\sum \lambda_i$", color="gray")
#plt.show()
fig.savefig(f"{folder}_overload_testbed.png")


y2 = np.load('offload_train_salmut.npy')
y1 = np.load('offload_baseline.npy')
#y1 = savgol_filter(y1[0:999], 11, 3)
b1 = savgol_filter(y1[0:999, 0], 21, 4)
b2 = savgol_filter(y1[0:999, 1], 21, 4)
b3 = savgol_filter(y1[0:999, 2], 21, 4)
s1 = savgol_filter(y2[0:999, 0], 21, 4)
s2 = savgol_filter(y2[0:999, 1], 21, 4)
s3 = savgol_filter(y2[0:999, 2], 21, 4)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(x, s2, '#377eb8', label='SALMUT')
ax.fill_between(x, s1, s3, color='#377eb8', alpha=0.2)
ax.plot(x, b2, '#f781bf', label='Baseline')
ax.fill_between(x, b1, b3, color='#f781bf', alpha=0.2)
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel('Timesteps', fontsize=14)
ax.set_ylabel('Offloading Count', fontsize=14)
ax.set_ylim([0, 90])
ax.legend(loc=1)
# ax.legend(labels, handles, bbox_to_anchor=(0., 1.02, 1., .102), prop={'size': 12}, loc='upper center',
#                   ncol=2, mode="expand", borderaxespad=0.)
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(x, lambd, color="gray", linestyle="dotted")
#ax2.plot(x, lambd, color="",marker="o")
ax2.set_ylabel(r"$\sum \lambda_i$", color="gray")
#plt.show()
fig.savefig(f"{folder}_offload_testbed.png")


"""
fig, ax = plt.subplots()
ax.plot(x, ov1[:,1], 'r-', label='PPO')
ax.fill_between(x, ov1[:,0], ov1[:,2], color='r', alpha=0.2)
ax.plot(x, ov2[:,1], 'b-', label='A2C')
ax.fill_between(x, ov2[:,0], ov2[:,2], color='b', alpha=0.2)
ax.plot(x, ov3[:,1], 'g-', label='SALMUT')
ax.fill_between(x, ov3[:,0], ov3[:,2], color='g', alpha=0.2)
ax.plot(x, ov4[:,1], 'c--', label='Policy Iteration')
ax.fill_between(x, ov4[:,0], ov4[:,2], color='c', alpha=0.2)
ax.plot(x, ov5[:,1], 'm--', label='Threshold=18')
ax.fill_between(x, ov5[:,0], ov5[:,2], color='m', alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Overloaded State per 1000 timesteps')
plt.legend()
fig.savefig(f"{folder}_overload.png")


fig, ax = plt.subplots()
ax.plot(x, off1[:,1], 'r-', label='PPO')
ax.fill_between(x, off1[:,0], off1[:,2], color='r', alpha=0.2)
ax.plot(x, off2[:,1], 'b-', label='A2C')
ax.fill_between(x, off2[:,0], off2[:,2], color='b', alpha=0.2)
ax.plot(x, off3[:,1], 'g-', label='SALMUT')
ax.fill_between(x, off3[:,0], off3[:,2], color='g', alpha=0.2)
ax.plot(x, off4[:,1], 'c--', label='Policy Iteration')
ax.fill_between(x, off4[:,0], off4[:,2], color='c', alpha=0.2)
ax.plot(x, off5[:,1], 'm--', label='Threshold=18')
ax.fill_between(x, off5[:,0], off5[:,2], color='m', alpha=0.2)
plt.xlabel('Timesteps')
plt.ylabel('Requests Offloaded per 1000 timesteps')
plt.legend()
fig.savefig(f"{folder}_offload.png")
"""
