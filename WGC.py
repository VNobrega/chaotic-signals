import numpy as np
import matplotlib.pyplot as plt


initValues = np.array([0.1218, 0.3499])
n = 5000
kpos = 100
sigma2 = 0.1
initValuesW = np.array([0.2457, 0.1640])


kAxis = np.arange(-kpos, kpos+1)

T = np.zeros([n, 2])
T2 = np.zeros([n, 2])
autCorrT = np.zeros([kpos*2+1, 2])
autCorrW = np.zeros([kpos*2+1, 2])




def tent(old):
    if(old < 0.5):
        new = 2*old
    else:
        new = 2*(1-old)
    return new + 1e-15*np.random.randn()

def genT(initValues, n):
    T = np.zeros([n,2])
    T[0,:] = initValues[:]
    for i in range(n-1):
        T[i+1,0] = tent(T[i,0]) 
        T[i+1,1] = tent(T[i,1]) 
        print(T[i+1,0], T[i+1,1])
    return T

def autocorrelation(series, kpos):
    r = np.zeros([kpos+1])
    r_total = np.zeros([kpos*2+1])
    for i in range(kpos+1):
        for j in range (len(series)-i):
            r[i]+=series[j]*series[j+i]
        r[i] = r[i]/(len(series)-i)        
    r_total = np.concatenate((r[:0:-1], r))
    return r_total

def Gtransform(series, sigma2):
    G = np.zeros((len(series), 2))
    for i in range(len(series)):
        r = np.sqrt(2*sigma2*np.log(1/(1-series[i,0])))
        theta = 2*np.pi*series[i,1]
        G[i,0] = r*np.cos(theta)
        G[i,1] = r*np.sin(theta)
    return G

def GtransformInv(series, sigma2):
    Ginv = np.zeros(2)
    r = (series[0]**2+series[1]**2)/(2*sigma2)
    Ginv[0] = 1-np.exp(-r)
    Ginv[1] = (1/(2*np.pi))*np.arctan2(series[1],series[0])
    return Ginv
            

def genW(initValues, sigma2, n):
    W = np.zeros([n,2])
    W[0,:] = initValues[:]
    for i in range(n-1):
        R = GtransformInv(W[i, :], sigma2)
        R_next = np.array([tent(R[0]), tent(R[1])])
        W[i+1, :] = Gtransform(R_next.reshape(1,2), sigma2)
    return W


T = genT(initValues, n)


W = genW(initValuesW, sigma2, n)



fig, signalT = plt.subplots(1, 2)
signalT[0].plot(T[:, 0])
signalT[0].set_title("Y1")
signalT[0].set_xlabel("n")
signalT[0].set_ylabel("y")
signalT[1].plot(T[:, 1])
signalT[1].set_title("Y2")
signalT[1].set_xlabel("n")
signalT[1].set_ylabel("y")

fig, signalTAuto = plt.subplots(1,2)
autCorrT[:, 0] = autocorrelation(T[:,0], kpos)
autCorrT[:, 1] = autocorrelation(T[:,1], kpos)
signalTAuto[0].plot(kAxis, autCorrT[:, 0])
signalTAuto[1].plot(kAxis, autCorrT[:, 1])
signalTAuto[0].set_ylabel("Ry1y1")
signalTAuto[1].set_ylabel("Ry2y2")
signalTAuto[0].set_xlabel("k")
signalTAuto[1].set_xlabel("k")



fig, signalW = plt.subplots(1, 2)
signalW[0].plot(W[:,0])
signalW[0].set_title("X1")
signalW[0].set_xlabel("n")
signalW[0].set_ylabel("x")
signalW[1].plot(W[:,1])
signalW[1].set_title("X2")
signalW[1].set_xlabel("n")
signalW[1].set_ylabel("x")

fig, signalWAuto = plt.subplots(1,2)
autCorrW[:, 0] = autocorrelation(W[:,0], kpos)
autCorrW[:, 1] = autocorrelation(W[:,1], kpos)
signalWAuto[0].plot(kAxis, autCorrW[:, 0])
signalWAuto[1].plot(kAxis, autCorrW[:, 1])
signalWAuto[0].set_ylabel("Rx1x1")
signalWAuto[1].set_ylabel("Rx2x2")
signalWAuto[0].set_xlabel("k")
signalWAuto[1].set_xlabel("k")



plt.show()