import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n = 10000
sigma2 = 0.1

initValuesT = np.array([0.1218, 0.3499])
initValuesW = np.array([0.2457, 0.1640])

lags = 10


autCorrT = np.zeros([2,lags*2+1])
autCorrW = np.zeros([2, lags*2+1])
kAxis = np.arange(-lags, lags+1)

def tent(x_u):
    xquadratic = (1-np.cos(np.pi*x_u))/2
    xquadraticnovo = 4*xquadratic*(1-xquadratic)
    xnovo = np.arccos(1-2*xquadraticnovo)/np.pi
    return xnovo

def atan2_0_to_2pi(y, x):
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2 * np.pi
    return angle


def genT(initValues, n):
    T = np.zeros([2,n])
    T[:,0] = initValues
    for i in range(n-1):
        T[0,i+1] = tent(T[0,i]) 
        T[1,i+1] = tent(T[1,i]) 
    return T

def autocorrelation(series, lags):
    r = np.zeros([lags+1])
    r_total = np.zeros([lags*2+1])
    for i in range(lags+1): 
        for j in range (len(series)-i):
            r[i]+=series[j]*series[j+i]
        r[i] = r[i]/(len(series)-i)        
    r_total = np.concatenate((r[:0:-1], r))
    return r_total

def Gtransform(series, sigma2):
    r = np.sqrt(2*sigma2*np.log(1/(1-series[0])))
    theta = 2*np.pi*series[1]
    G = np.array([r*np.cos(theta), r*np.sin(theta)])
    return G


def GtransformInv(series, sigma2):
    Ginv = np.zeros(2)
    r = (series[0]**2+series[1]**2)/(2*sigma2)
    Ginv[0] = 1-np.exp(-r)
    Ginv[1] = (1/(2*np.pi))*atan2_0_to_2pi(series[1],series[0])
    return Ginv
            

def genW(initValues, sigma2, n):
    W = np.zeros([2,n])
    W[:,0] = initValues[:]
    for i in range(n-1):
        R = GtransformInv(W[:, i], sigma2)
        R_next = np.array([tent(R[0]), tent(R[1])])
        W[:,i+1] = Gtransform(R_next, sigma2)
    return W


W = genW(initValuesW, sigma2, n)
T =  genT(initValuesT, n)


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

autCorrT[0,:] = autocorrelation(T[0,:], lags)
autCorrT[1,:] = autocorrelation(T[1,:], lags)

ax[2,0].stem(kAxis, autCorrT[0,:])
ax[2,0].set_ylabel("Ry1y1")
ax[2,0].set_xlabel("k")

ax[2,1].stem(kAxis, autCorrT[1,:])
ax[2,1].set_ylabel("Ry2y2")
ax[2,1].set_xlabel("k")

plt.tight_layout()


fig, ax = plt.subplots(3, 2, figsize=(10,8))

ax[0,0].stem(W[0,:])
ax[0,0].set_title("X1")
ax[0,0].set_xlabel("n")
ax[0,0].set_ylabel("x")

ax[0,1].stem(W[1,:])
ax[0,1].set_title("X2")
ax[0,1].set_xlabel("n")
ax[0,1].set_ylabel("x")

ax[1,0].hist(W[0,:], density=True, bins='fd')
ax[1,0].set_title("Histogram X1")
ax[1,0].set_xlim(-3,3)

ax[1,1].hist(W[1,:], density=True, bins='fd')
ax[1,1].set_title("Histogram X2")
ax[1,1].set_xlim(-3,3)

eixo = np.linspace(-3,3,1000)

ax[1,0].plot(eixo, norm.pdf(eixo, loc=0, scale=np.sqrt(sigma2)), 'r-', lw=2)
ax[1,1].plot(eixo, norm.pdf(eixo, loc=0, scale=np.sqrt(sigma2)), 'r-', lw=2)

autCorrW[0,:] = autocorrelation(W[0,:], lags)
autCorrW[1,:] = autocorrelation(W[1,:], lags)

ax[2,0].stem(kAxis, autCorrW[0,:])
ax[2,0].set_ylabel("Rx1x1")
ax[2,0].set_xlabel("k")

ax[2,1].stem(kAxis, autCorrW[1,:])
ax[2,1].set_ylabel("Rx2x2")
ax[2,1].set_xlabel("k")

plt.tight_layout()


plt.show()


