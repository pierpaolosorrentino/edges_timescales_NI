from sklearn.metrics import mutual_info_score
import numpy as np
import matplotlib.pyplot as plt  
from scipy.signal import argrelextrema

# MUTUAL INFORMATION SCORE
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# PARTIAL AUTO CORRELATION
def Pautocorr(x,lags):
    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)

#Here we estimate the time lag as the first minimum of the average mutual information 
#    score between the time series and the time lagged one.
#Since there is evidence of a second slower time scale we select another minimum as 
#    a second time lag value.
def TauLag(signal, tau_max, n_nodes, method, nbin=100):
    tau_fast=tau_max
    if method=='MI':
        Measure = np.zeros((tau_max,n_nodes))
        for tau in np.arange(1, tau_max):
            unlagged = signal[:-tau,:]
            lagged = np.roll(signal, -tau, axis=0)[:-tau,:]
            for n in range(n_nodes):
                Measure[tau-1,n]=calc_MI(unlagged[:,n], lagged[:,n], nbin)
            Amis=np.mean(np.asarray(Measure),axis=1)[:-1]
        # for local minima
        #tau_fast=argrelextrema(Amis, np.less)[0][0];#tau_slow=np.argmin(Amis)
        limite=np.mean(Amis)+np.std(Amis)
        #plt.figure(figsize=(18,4))
        #plt.plot(Measure,'-',alpha=0.15);plt.grid()
        #plt.plot(Amis,'k-',linewidth=3);plt.ylim((np.min(Amis),limite));plt.xlim((0,tau_max)); 
        #plt.title(('Delay estimate via average Mutual Information')) #: Tau_fast at %d'%tau_fast)))
    elif method=='PAC':
        plt.figure(figsize=(18,6))
        lags=np.arange(tau_max)
        Measure=np.zeros((len(lags), n_nodes))
        for r in range(n_nodes):
            Measure[:,r]=Pautocorr(signal[:,r],lags)
            if list(np.where(np.diff(np.sign(Measure[:,r])))[0]):
                inv=np.where(np.diff(np.sign(Measure[:,r])))[0][0]
            else:
                inv=tau_max
            if inv<tau_fast:
                tau_fast=inv
            plt.plot(Measure[:,r],linewidth=0.7)
        plt.axvline(tau_fast);plt.xlabel('tau');plt.grid() #;plt.axvline(tau_slow);
        plt.title('Delay estimate via Partial Autocorrelation: Tau_fast at %d'%tau_fast);plt.xlim((0,tau_max))
    else:
        print('insert a measure method: method can be MI or PAC')
        
    return Measure

# Delay embedding function that returns the extended phase space.
def DelayEmbedding(signal, tau, N):
    unlagged=signal[:-tau*N,:]
    PS=np.expand_dims(unlagged,-1)
    for n in np.arange(1,N+1):
        lagged=np.roll(signal, -tau*n, axis=0)[:-tau*N,:]
        PS=np.dstack((PS, np.expand_dims(lagged,-1)))
    return PS

# Delay embedding function that returns the extended phase space.
# If grad!=0 we also add the derivative of the time series to the extended phase space
def DelayEmbeddingGrad(signal, tau, N):
    unlagged=signal[:-tau*N,:]
    PS=np.expand_dims(unlagged,-1)
    for n in np.arange(1,N+1):
        lagged=np.roll(signal, -tau*n, axis=0)[:-tau*N,:]
        PS=np.dstack((PS, np.expand_dims(lagged,-1)))
    PS=np.dstack((PS,np.expand_dims(np.gradient(unlagged,axis=1),-1)))
    for n in range(1,N+1):
        lagged=np.roll(signal, -tau*n, axis=0)[:-tau*N,:]
        PS=np.dstack((PS,np.expand_dims(np.gradient(lagged,axis=1),-1)))
    return PS

#def DelayEmbedding(signal, tau_fast, tau_slow, N):
#    unlagged=signal[:-tau_slow*N,:]
#    PS=unlagged.T 
#    for n in np.arange(1,N+1):
#        lagged_fast=np.roll(signal, -tau_fast*n, axis=0)[:-tau_slow*N,:]
#        PS=np.vstack((PS, lagged_fast.T))
#    return PS

# If grad!=0 we also add the derivative of the time series to the extended phase space
#def DelayEmbedding(signal, tau_fast, tau_slow, N):
#    unlagged=signal[:-N*tau_slow,:]
#    PS=unlagged.T 
#    for n in np.arange(1,N+1):
#        lagged_fast=np.roll(signal, -tau_fast*n, axis=0)[:-tau_slow*N,:]
#        PS=np.vstack((PS, lagged_fast.T))
#    for n in np.arange(1,N+1):
#        lagged_slow=np.roll(signal, -tau_slow*n, axis=0)[:-tau_slow*N,:]
#        PS=np.vstack((PS, lagged_slow.T))
#    return PS



