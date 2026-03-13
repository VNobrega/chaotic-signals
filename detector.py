import plot as plotfile
import numpy as np
from scipy.stats import norm

def time_reversibility_stat(x):
    d = np.diff(x)
    return np.mean(d**3)



surrogates = []



for _ in range(1000):
    noise = np.random.randn(plotfile.n)
    surrogates.append(time_reversibility_stat(noise))


signal_mean = np.array([time_reversibility_stat(plotfile.W[0,:]), time_reversibility_stat(plotfile.W[1,:])])
surrogates_mean = np.mean(surrogates)
surrogate_sd = np.std(surrogates)

z = np.array([(signal_mean[0]-surrogates_mean)/surrogate_sd,(signal_mean[1]-surrogates_mean)/surrogate_sd])
p_value = 2*(1 - norm.cdf(abs(z)))

print("signal: ", signal_mean[0],",", signal_mean[1])
print("noise mean: ", surrogates_mean)
print("noise std: ", surrogate_sd)
print("p: ", p_value[0],",", p_value[1])

