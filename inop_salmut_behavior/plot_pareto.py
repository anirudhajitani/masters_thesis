import matplotlib.pyplot as plt
import numpy as np
import sys
import os

folder = sys.argv[1]
os.chdir(f"./{folder}/results")

off_salmut=np.load('offload_med_salmut_combined.npy')
ov_salmut=np.load('overload_med_salmut_combined.npy')
off_baseline=np.load('offload_med_thres_eval_20.npy')
ov_baseline=np.load('overload_med_thres_eval_20.npy')

fig, ax = plt.subplots(figsize=(7,4))
plt.scatter(off_salmut[1,:], ov_salmut[1,:], color='#377eb8', label='salmut')
plt.scatter(off_baseline, off_baseline, color='#f781bf', label='baseline')
plt.xlabel('Offload Count')
plt.ylabel('Overload Count')
plt.legend()
fig.savefig(f'pareto_{folder}.png')
