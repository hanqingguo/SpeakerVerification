# This script takes WAV file, and output STFT figures, MFCCs figures

import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import dct

sns.set()
import os
import scipy
import librosa.display
from IPython.display import Audio
import random


# sample_rate = 48000
sample_rate = 192000
utter, sr = librosa.core.load('Nick-high/2.wav', sample_rate)  # load utterance audio
# intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
# for interval in intervals:
utter_part = utter  # save first and last 180 frames of spectrogram.
print(utter_part.shape)
S = librosa.core.stft(y=utter_part)
S = np.abs(S) ** 2
print("Size of S is {}".format(S.shape))
mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=2048, n_mels=128)
S = np.log10(np.dot(mel_basis, S) + 1e-6)
print(S.shape)
plt.rcParams["axes.grid"] = False
plt.imshow(S, cmap='hot', origin='lower')
pos, labels = plt.yticks()
pos = np.arange(0,128,step=128/7)
labels = np.arange(0,128,step=20)
xpos, xlabels = plt.xticks()
print("Original xpos is {}, original xlabels is {}".format(xpos, xlabels))
xpos = np.arange(0,xpos[-1], step=xpos[-1]/4)
xlabels = np.arange(0,2.5, step=0.5)

plt.xticks(xpos, xlabels)
plt.yticks(pos, labels)
# plt.title("GE2E Mel-Filter STFT : P1S1")
plt.xlabel("Seconds")
plt.ylabel("Mel-Filters")
# plt.show()
# plt.pause(2)
# Extract MFCC feature
# mfccs = librosa.feature.mfcc(y=utter_part.astype('float'), sr=sample_rate, n_mfcc=10)
mfccs = librosa.feature.mfcc(S = S, sr=sample_rate, n_mfcc=128 )
print("Shape of mfcc features is {}".format(mfccs.shape))
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', hop_length=58)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
# plt.savefig('Nick-hisiri-high-useS-100')
plt.show()


M = dct(S, axis=0, type=2, norm='ortho')
M1 = M[:40]
print(M.shape, M1.shape)

