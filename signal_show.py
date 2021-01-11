import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

files = pd.read_pickle('data/Pat1_1/Pat1Train_121_1.pkl')

signal = np.array(files['ch1'])

t = np.arange(0, len(signal))
plt.plot(t, signal)
plt.show()
