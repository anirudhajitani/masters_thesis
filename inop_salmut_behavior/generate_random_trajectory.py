import numpy as np
import random

random.seed(20)
traj = []
for i in range(0, int(1e6)):
    traj.append(random.uniform(0,1))

np.save('event_traj.npy', traj)
