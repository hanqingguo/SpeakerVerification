# This script takes WAV file, and output STFT figures, MFCCs figures

import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import scipy
import librosa.display
from IPython.display import Audio
import random

########################################################################
# Original feature of GE2E
########################################################################

utter, sr = librosa.core.load('SA1.WAV', 16000)  # load utterance audio
#utter, sr = librosa.core.load('SA1.WAV', 16000)
#intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
#for interval in intervals:
utter_part = utter  # save first and last 180 frames of spectrogram.
print(utter_part.shape)
S = librosa.core.stft(y=utter_part, n_fft=512,
                      win_length=int(0.025 * sr), hop_length=int(0.01 * sr))
S = np.abs(S) ** 2
mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
S = np.log10(np.dot(mel_basis, S) + 1e-6)
print(S.shape)
plt.rcParams["axes.grid"] = False
plt.imshow(S, cmap='hot', origin='lower')
pos, labels = plt.yticks()
pos = np.arange(0,45,step=40/6)
xpos, xlabels = plt.xticks()
xpos = np.arange(0,xpos[-1], step=xpos[-1]/5)
xlabels = np.arange(0,utter_part.shape[0]/sr, step=0.5)

plt.xticks(xpos, xlabels)
plt.yticks(pos, np.arange(0, 9600, step=1600))
plt.title("GE2E Mel-Filter STFT : P1S1")
plt.xlabel("Seconds")
plt.ylabel("Hz")
plt.show()

mfccs = librosa.feature.mfcc(y=utter_part.astype('float'), sr=16000, n_mfcc=20)
print("Shape of mfcc features is {}".format(mfccs.shape))
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()