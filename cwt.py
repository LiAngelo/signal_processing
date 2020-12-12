
__author__ = "Yunzhong Li"
__maintainer__ = "Yunzhong Li"
__version__ = "1.0.1"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pywt
import glob
import os
import matplotlib.colors as colors

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


def cwt(signal, wavename='cgau8', totalscal=201, sampling_rate=400):
    '''do continuous wavelet transform

    Parameters
    ----------
    signal: {2d numpy array}
        The input signal, 200 x 240000.
    wavename: {string}
        The wave selected to transform signal.
    totalscal: {int}
        different scales corresponding different frequency bands need>200, set:201
    sampling_rate:{int}
        The sampling rate of signal, set:400Hz
    '''

    fc = pywt.central_frequency(wavename)    # central frequency
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1) # caculate scales
    cwt_signal, frequencies = pywt.cwt(signal, scales, wavename, 1.0 / sampling_rate)
    return np.abs(cwt_signal), frequencies

if __name__ == '__main__':
    files = glob.glob1('data/', '*.pkl')

    for i in range(len(files)):
        file_name = files[i]
        segment_no, label = file_name[10:-4].split('_')
        df = pd.read_pickle(os.path.join('data', file_name))
        sampling_rate = 400
        t = np.arange(0, 30, 1.0 / sampling_rate)
        signal = df['ch13']

        #cwt
        cwt_signal, frequencies = cwt(signal)

        #normalization
        norm_signal = [[0 for i in range(cwt_signal.shape[1])] for i in range(len(cwt_signal))]
        for i in range(len(cwt_signal)):
            norm_signal[i] = normalization(cwt_signal[i, :])
        norm_signal = np.array(norm_signal)

        #split into 20 30sec window
        for i in range(20):
            plt.figure(figsize=(8, 4))
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            Z = norm_signal[:, i * 12000:(i + 1) * 12000]
            plt.pcolormesh(t, frequencies, Z, vmin=Z.min(), vmax=Z.max())
            #plt.pcolormesh(t, frequencies, Z, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))
            plt.axis('off')
            name = str(segment_no) + '_' + str(i) + '_' + str(label)
            plt.savefig((os.path.join('./image_cwt',name+'.png')))
            #plt.show()
