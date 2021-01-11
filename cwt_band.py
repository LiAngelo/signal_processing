__author__ = "Yunzhong Li"
__maintainer__ = "Yunzhong Li"
__version__ = "1.0.1"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import pywt
import glob
import os

PSD_FREQ = np.array([[0, 4], [4, 8], [8, 12], [12, 30], [30, 70], [70, 200]])
sampling_rate = 400


def split_signal(data, epoch_length_sec, stride_sec):
    ''' split 10 minutes into epochs
    Parameters
    ----------
    data: {2d numpy array: channels * samples}
        The input signal, 16 x 240000.
    epoch_length_sec: int
        The length (sec) of each epoch.
    stride_sec: int
        The length (sec) of stride.
        epoch_length_sec == stride_sec mean no overlap
    '''
    signal = np.array(data, dtype=np.float32)
    signal_epochs = []
    samples_in_epoch = epoch_length_sec * sampling_rate
    stride = stride_sec * sampling_rate

    # compute dropout indices
    drop_indices_c0 = np.where(signal[:, 0] == signal[:, 1])[0]
    drop_indices_c1 = np.where(signal[:, 14] == signal[:, 15])[0]
    drop_indices = np.intersect1d(drop_indices_c0, drop_indices_c1)
    drop_indices = np.append(drop_indices, len(signal))

    window_start = 0
    for i in drop_indices:
        epoch_start = window_start
        epoch_end = epoch_start + samples_in_epoch

        while epoch_end < i:
            signal_epochs.append(signal[epoch_start:epoch_end, :])
            epoch_start += stride
            epoch_end += stride

        window_start = i + 1

    return np.array(signal_epochs)


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

    fc = pywt.central_frequency(wavename)  # central frequency
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)  # caculate scales
    cwt_signal, frequencies = pywt.cwt(signal, scales, wavename, 1.0 / sampling_rate)
    return np.abs(cwt_signal), frequencies


if __name__ == '__main__':
    files = glob.glob1('data/Pat1_0/', '*.pkl')

    for i in range(len(files)):
        file_name = files[i]
        segment_no, label = file_name[10:-4].split('_')
        df = pd.read_pickle(os.path.join('data/Pat1_0/', file_name))
        signal = df.loc[:, 'ch0':'ch15']

        # signal split
        signal = split_signal(signal, 30, 30)

        if len(signal.shape) == 3:
            # signal with channel0
            signal_c0 = signal[:, :, 0]

            # cwt
            cwt_signal, frequencies = cwt(signal_c0)

            for epoch in range(cwt_signal.shape[1]):
                ret = []
                for freq_band in PSD_FREQ:
                    tmp = (frequencies >= freq_band[0]) & (frequencies < freq_band[1])
                    ret.append((cwt_signal[tmp, epoch, :].mean(0)))
                ret = np.log10(np.array(ret) / np.sum(ret, axis=0))

                # log plot
                plt.figure(figsize=(2.56, 2.56))
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                t = np.arange(0, ret.shape[1])
                BAND = np.array(['0', 'Delta', 'Theta', 'Alpha', 'Beta', 'low-gamma', 'high-gamma'])
                plt.pcolormesh(t, BAND, np.float32(ret), norm=colors.Normalize(vmin=-2, vmax=0))
                plt.axis('off')
                name = str(segment_no) + '_' + str(epoch) + '_' + str(1) + '_' + str(label)
                plt.savefig((os.path.join('./image/band/Pat1_30sec_0', name + '.jpeg')))
