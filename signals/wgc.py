import numpy as np


def tent(x_u):
    xquadratic = (1-np.cos(np.pi*x_u))/2
    xquadraticnovo = 4*xquadratic*(1-xquadratic)
    xnovo = np.arccos(1-2*xquadraticnovo)/np.pi
    #xnovo = np.where(x_u < 0.5, 2*x_u, 2*(1-x_u))+np.random.rand()*1e-15
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

def autocorrelation(series, k):
    series = series - np.mean(series)
    r = np.correlate(series, series, mode='full')
    N = len(series)
    r = r[N-1-k:N+k]
    r = r / r[k]
    return r



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




