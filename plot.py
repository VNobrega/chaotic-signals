import WGC
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 10000
sigma2 = 1
lags = 10
W = WGC.genW([0.443, 0.763], sigma2, n)
T =  WGC.genT(np.random.rand(2), n)




# Tent map

autCorrT = np.zeros([2,lags*2+1])
autCorrW = np.zeros([2, lags*2+1])
kAxis = np.arange(-lags, lags+1)


plt.rcParams.update({
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24
})

'''
fig, ax = plt.subplots(3, 2, figsize=(10,8))

ax[0,0].stem(T[0,:])
ax[0,0].set_title("Y1")
ax[0,0].set_xlabel("n")
ax[0,0].set_ylabel("y")

ax[0,1].stem(T[1,:])
ax[0,1].set_title("Y2")
ax[0,1].set_xlabel("n")
ax[0,1].set_ylabel("y")

ax[1,0].hist(T[0,:], density=True, bins='fd')
ax[1,0].set_title("Histogram Y1")

ax[1,1].hist(T[1,:], density=True, bins='fd')
ax[1,1].set_title("Histogram Y2")

autCorrT[0,:] = WGC.autocorrelation(T[0,:], lags)
autCorrT[1,:] = WGC.autocorrelation(T[1,:], lags)

ax[2,0].stem(kAxis, autCorrT[0,:])
ax[2,0].set_ylabel("Ry1y1")
ax[2,0].set_xlabel("k")

ax[2,1].stem(kAxis, autCorrT[1,:])
ax[2,1].set_ylabel("Ry2y2")
ax[2,1].set_xlabel("k")

plt.tight_layout()

'''

#WGC

fig, ax = plt.subplots(3, 2, figsize=(12, 12))

# --- Time series ---
ax[0,0].stem(W[0,:])
ax[0,0].set_xlabel("n")
ax[0,0].set_ylabel(r"(a) $w_1(n)$")

ax[0,1].stem(W[1,:])
ax[0,1].set_xlabel("n")
ax[0,1].set_ylabel(r"(b) $w_2(n)$")

# --- Histograms ---
ax[1,0].hist(W[0,:], density=True, bins='fd')
ax[1,0].set_xlim(-3,3)
ax[1,0].set_xlabel("Amplitude $x_1$")
ax[1,0].set_ylabel(r"(c) Densidade de $x_1$")

ax[1,1].hist(W[1,:], density=True, bins='fd')
ax[1,1].set_xlim(-3,3)
ax[1,1].set_xlabel("Amplitude $x_2$")
ax[1,1].set_ylabel(r"(d) Densidade $x_2$")

eixo = np.linspace(-3,3,1000)

ax[1,0].plot(eixo, norm.pdf(eixo, loc=0, scale=np.sqrt(sigma2)), 'r-', lw=2)
ax[1,1].plot(eixo, norm.pdf(eixo, loc=0, scale=np.sqrt(sigma2)), 'r-', lw=2)

# --- Autocorrelation ---
autCorrW[0,:] = WGC.autocorrelation(W[0,:], lags)
autCorrW[1,:] = WGC.autocorrelation(W[1,:], lags)

ax[2,0].stem(kAxis, autCorrW[0,:])
ax[2,0].set_ylabel(r"(e) R$w_1$$w_1$(k)")
ax[2,0].set_xlabel("k")

ax[2,1].stem(kAxis, autCorrW[1,:])
ax[2,1].set_ylabel(r"(f) R$w_2$$w_2$(k)")
ax[2,1].set_xlabel("k")

plt.tight_layout()

'''

rand_vec = np.random.randn(n)

fig_rand, ax_rand = plt.subplots(figsize=(6,4))

ax_rand.hist(rand_vec, density=True, bins='fd')
ax_rand.set_title("Histogram randn")
ax_rand.set_xlim(-3,3)
eixo = np.linspace(-3,3,1000)
ax_rand.plot(eixo, norm.pdf(eixo, loc=0, scale=1), 'r-', lw=2)

plt.tight_layout()
'''


plt.subplots_adjust(hspace=1)
plt.show()
