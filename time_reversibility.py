import plot as plotfile
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def time_reversibility_stat(x):
    d = np.diff(x)
    return np.mean(d**3)



surrogates = []



for _ in range(100):
    noise = np.random.randn(plotfile.n)
    surrogates.append(time_reversibility_stat(noise))


signal_mean = np.array([time_reversibility_stat(plotfile.W[0,:]), time_reversibility_stat(plotfile.W[1,:])])
surrogates_mean = np.mean(surrogates)
surrogate_sd = np.std(surrogates, ddof=1)

z = np.array([(signal_mean[0]-surrogates_mean)/surrogate_sd,(signal_mean[1]-surrogates_mean)/surrogate_sd])
p_value = 2*(1 - norm.cdf(abs(z)))

print("noise_mean: {:.5f}".format(surrogates_mean))
print("noise_std: {:.5f}".format(surrogate_sd))
print("signal: {:.5f}".format(signal_mean[0]),",{:.5f}".format(signal_mean[1]))
print("p: {:.3f}, {:.3f}".format(p_value[0], p_value[1]))

plt.show()