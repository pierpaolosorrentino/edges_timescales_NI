import numpy as np
import matplotlib.pyplot as plt  
import collections as cl
from scipy.signal import butter, lfilter, freqz, hilbert, chirp, argrelextrema

def butter_bandpass(band, fs, FreqBands_dict, order):
    lowcut=FreqBands_dict[band][0]
    highcut=FreqBands_dict[band][1]
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, band, fs, FreqBands_dict, order):
    b, a = butter_bandpass(band, fs, FreqBands_dict, order=order)
    y = lfilter(b, a, data)
    return y

def DefineBands(n_inter=25,inter=4,order=3):
    #Homogeneous partition of frequencies stored into a dict
    Frequency_Intervals=dict((i, [i*inter+0.01,i*inter+inter+0.01, order]) for i in np.arange(n_inter))
    return Frequency_Intervals

#Standard frequency bands
Frequency_Bands = { 'delta' : [0.01,4.,3], 'theta' : [4.,7.,5], 'alpha' : [8.,12,5], 'beta' : [12.,30.,7], 'gamma' : [30.,99.,8]} 
Frequency_Bands = cl.OrderedDict(Frequency_Bands) 

def SplitBands(signal, fs, FreqBands_dict):
    BandSignals=np.zeros((len(FreqBands_dict.keys()),len(signal),len(signal[0])))
    for b, band in enumerate(FreqBands_dict.keys()):
        order = FreqBands_dict[band][2]
        for r in range(len(signal[0])):
            BandSignals[b,:,r] = butter_bandpass_filter(signal[:,r], band, fs, FreqBands_dict, order=order)
    return BandSignals         
            
def HilbertSignal(signal, fs):
    #signal~(time,regions)
    Envelope=np.zeros((len(signal),len(signal[0])))
    InstantPhase=np.zeros((len(signal),len(signal[0])))
    InstantFreq=np.zeros((len(signal)-1,len(signal[0])))
    for r in range(len(signal[0])):
        analytic_signal = hilbert(signal[:,r])
        Envelope[:,r] = np.abs(analytic_signal)
        phases= np.unwrap(np.angle(analytic_signal))
        InstantPhase[:,r]=phases #( phases + np.pi) % (2 * np.pi ) - np.pi
        #InstantFreq[:,r] = (np.diff(InstantPhase[:,r])/(2.0*np.pi) * fs)
        
    return Envelope, InstantPhase #, InstantFreq