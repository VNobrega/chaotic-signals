from metrics import knn
from metrics import reversibility
from signals import wgc
import numpy as np
from scipy.stats import norm


n = 10000
sigma2 = 1
W = wgc.genW([0.443, 0.763], sigma2, n)


surrogates = []
for _ in range(100):
    noise = np.random.randn(n)
    surrogates.append(knn.nonlinear_prediction_rmse(noise))

signal_rmse = np.array([knn.nonlinear_prediction_rmse(W[0,:]), knn.nonlinear_prediction_rmse(W[1,:])])
surrogates_mean = np.mean(surrogates)
surrogate_sd = np.std(surrogates, ddof=1)

z = np.array([(signal_rmse[0]-surrogates_mean)/surrogate_sd,(signal_rmse[1]-surrogates_mean)/surrogate_sd])
p_value = 2*(1 - norm.cdf(abs(z)))

print("k-NN:")
print("noise_mean RMSE: {:.5f}".format(surrogates_mean))
print("noise_std RMSE: {:.5f}".format(surrogate_sd))
print("signal RMSE: {:.5f}".format(signal_rmse[0]),",{:.5f}".format(signal_rmse[1]))
print("p: {:.3f}, {:.3f}".format(p_value[0], p_value[1]))
print("\n")


surrogates = []
for _ in range(100):
    noise = np.random.randn(n)
    surrogates.append(reversibility.time_reversibility_stat(noise))


signal_rev = np.array([reversibility.time_reversibility_stat(W[0,:]), reversibility.time_reversibility_stat(W[1,:])])
surrogates_mean = np.mean(surrogates)
surrogate_sd = np.std(surrogates, ddof=1)

z = np.array([(signal_rev[0]-surrogates_mean)/surrogate_sd,(signal_rev[1]-surrogates_mean)/surrogate_sd])
p_value = 2*(1 - norm.cdf(abs(z)))

print("Resultados baseados no terceiro momento:")
print("noise_mean third moment: {:.5f}".format(surrogates_mean))
print("noise_std third moment: {:.5f}".format(surrogate_sd))
print("signal  third moment: {:.5f}".format(signal_rev[0]),",{:.5f}".format(signal_rev[1]))
print("p: {:.3f}, {:.3f}".format(p_value[0], p_value[1]))