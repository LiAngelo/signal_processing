import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import stft
import glob
import os

def normalization(signal):
    '''normaliz each frequency band to 0-1 range separately

    Parameters
    ----------
    signal: {2d numpy array}
        The input signal, 200 x 240000.
    '''
    minVals = min(signal)
    maxVals = max(signal)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(signal))
    m = signal.shape
    normData = signal - np.tile(minVals, m)
    normData = normData/np.tile(ranges, m)
    return normData

def stft_transform(signal, nperseg=400,noverlap=1, fs=400):
    '''do short-time fourier transform

    Parameters
    ----------
    signal: {2d numpy array}
        The input signal, 200 x 240000.
    nperseg: {int}
        The length of each window, due to 400hz sampling rate, 400 points length to 1sec
    noverlap: {int}
        Set noverlap = 1. No noverlap
    fs: {int}
        The sampling rate of signal, set:400Hz
    '''
    f, t, zxx = stft(signal, nperseg=nperseg, noverlap=noverlap, fs=fs)
    return f, t, np.abs(zxx)

if __name__ == '__main__':
    files = glob.glob1('data/', '*.pkl')

    for i in range(len(files)):
        file_name = files[i]
        segment_no, label = file_name[10:-4].split('_')
        df = pd.read_pickle(os.path.join('data', file_name))
        sampling_rate = 400
        signal = df['ch13']

        # stft
        f, t, zxx = stft_transform(signal)

        # normalization
        norm_signal = [[0 for i in range(zxx.shape[1])] for i in range(len(zxx))]
        for i in range(len(zxx)):
            norm_signal[i] = normalization(zxx[i, :])
        norm_signal = np.array(norm_signal)

        # split into 20 30sec window
        for i in range(20):
            plt.figure(figsize=(8, 2))
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.pcolormesh(t[i*30:(i+1)*30], f, norm_signal[:,i*30:(i+1)*30])
            plt.axis('off')
            name = str(segment_no) + '_' + str(i) + '_' + str(label)
            plt.savefig((os.path.join('./image_stft',name+'.png')))
