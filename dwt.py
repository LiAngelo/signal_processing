import numpy as np
import  matplotlib.pyplot as plt
import pywt
import pandas as pd

iter_freqs = [
    {'name':'Delta', 'fmin':0, 'fmax':4},
    {'name':'Theta', 'fmin':4, 'fmax':8},
    {'name':'Alpha', 'fmin':8, 'fmax':12},
    {'name':'Beta', 'fmin':12, 'fmax':30},
    {'name':'low-gamma', 'fmin':30, 'fmax':70},
    {'name':'high-gamma', 'fmin':70, 'fmax':200},
]

df = pd.read_pickle('data/Pat1Train_1_1.pkl')
signal_400hz = df['ch13']

maxlevel = 8
fs=400
wp = pywt.WaveletPacket(data=signal_400hz, wavelet='db4', mode='symmetric', maxlevel=maxlevel)
freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
freqBand = fs/(2**maxlevel)
print()