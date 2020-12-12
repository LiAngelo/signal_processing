

__author__ = "Yunzhong Li"
__maintainer__ = "Yunzhong Li"
__version__ = "1.0.1"

import numpy as np
import pandas as pd
import pywt
import glob
import os

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
    min_band = np.ones((16, 200), dtype=float)
    max_band = np.zeros((16, 200), dtype=float)

    for file in range(len(files)):
        file_name = files[file]
        df = pd.read_pickle(os.path.join('data', file_name))
        signal = df.loc[:, 'ch0':'ch15']
        signal = np.array(signal)

        for channel in range(signal.shape[1]):
            cwt_signal, frequencies = cwt(signal[:, channel])
            for i in range(len(frequencies)):
                if min_band[channel, i] > np.min(cwt_signal[i]):
                    min_band[channel, i] = np.min(cwt_signal[i])
                if max_band[channel, i] < np.max(cwt_signal[i]):
                    max_band[channel, i] = np.max(cwt_signal[i])
        print(file)

    print(min_band)
    print(max_band)
    np.savetxt('./min.txt', min_band)
    np.savetxt('./max.txt', max_band)