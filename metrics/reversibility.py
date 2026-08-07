import numpy as np

def time_reversibility_stat(x):
    d = np.diff(x)
    return np.mean(d**3)


