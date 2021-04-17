import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import random
from collections import Counter

folder = sys.argv[1]
os.chdir(f"./{folder}/results")

off_salmut=np.load('offload_train_salmut.npy')
ov_salmut=np.load('overload_train_salmut.npy')
off_baseline=np.load('offload_baseline.npy')
ov_baseline=np.load('overload_baseline.npy')
x = [i for i in range(34)]

off_salmut = list(off_salmut[:,1])
ov_salmut = list(ov_salmut[:,1])
off_baseline = list(off_baseline[:,1])
ov_baseline = list(ov_baseline[:,1])


fig, ax = plt.subplots(figsize=(7,4))
combos = list(zip(off_salmut, ov_salmut))
weight_counter = Counter(combos)
weights = [weight_counter[(off_salmut[i], ov_salmut[i])] for i in range(999)]
plt.scatter(off_salmut, ov_salmut, s=weights, color='#377eb8', label='SALMUT')
combos = list(zip(off_baseline, off_baseline))
weight_counter = Counter(combos)
weights = [weight_counter[(off_baseline[i], off_baseline[i])] for i in range(999)]
plt.scatter(off_baseline, off_baseline, s=weights, color='#f781bf', label='Baseline')
plt.xlim(0,100)
plt.ylim(0,35)
plt.xlabel('Offload Count')
plt.ylabel('Overload Count')
plt.legend()
plt.show()
fig.savefig(f'pareto_weight_testbed_{folder}.png')
#fig.savefig(f'pareto_testbed_{folder}.png')

fig, ax = plt.subplots(figsize=(7,4))
plt.scatter(off_salmut[:,1], ov_salmut[:,1], color='#377eb8', label='SALMUT')
#plt.scatter(x, x, color='#f781bf', label='Baseline')
plt.scatter(off_baseline[:,1], off_baseline[:,1], color='#f781bf', label='Baseline')
#plt.scatter(off_salmut[1,:], ov_salmut[1,:], color='red', label='salmut')
#plt.scatter(off_baseline[1,:], off_baseline[1,:], color='blue', label='baseline')
plt.xlim(0,100)
plt.ylim(0,40)
plt.xlabel('Offload Count')
plt.ylabel('Overload Count')
plt.legend()
fig.savefig(f'pareto_testbed_{folder}.png')
