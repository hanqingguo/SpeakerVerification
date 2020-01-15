# This file contains person dataset selection,
# .wav file selection.
# mel-filters choose
# nmfcc features choose


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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as patches


def readfile(file):
    sr, samples = scipy.io.wavfile.read(file)
    # print("Person : {}, VoiceID : {}".format(os.path.dirname(file), file[-5:]))
    return sr, samples


def get_stft(sr, samples):
    X = librosa.stft(samples.astype('float'))
    Xdb = librosa.amplitude_to_db(np.abs(X))
    # print("Shape of STFT Original : {}".format(Xdb.shape))
    # Shape of Xdb: (1025, 752)
    # 1025 is determined by nfft, 752 is determined by higher time resolution of high smapling rate
    librosa.display.specshow(Xdb, sr=sr, x_axis='frames', y_axis='hz')
    plt.show()
    return Xdb


def gen_mel(sr, nfft, n_mels):
    # sample rate 太高导致mel filters 在高频处没有响应
    mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=n_mels)
    # print("Shape of Mel Filters : {}".format(mel_basis.shape))
    librosa.display.specshow(mel_basis, x_axis='linear',sr=sr)
    # plt.title("Mel Filters")
    # plt.show()
    return mel_basis


def get_mfcc(mel_basis, max_idx, Xdb, sr, n_mfcc, log=True):
    dot_result = np.dot(mel_basis, Xdb)
    if(log):
        dot_result = np.log10(dot_result + 1e-6)
    librosa.display.specshow(dot_result, sr=sr, x_axis='time', y_axis='hz')
    plt.show()

    mfccs = librosa.feature.mfcc(S = dot_result, sr=sr, n_mfcc=n_mfcc)
    fig, ax = plt.subplots(1)
    librosa.display.specshow(mfccs, x_axis='time')
    x_axis_res = 17.35/Xdb.shape[0]
    y_axis_res = 10/Xdb.shape[1]
    rect = patches.Rectangle((x_axis_res*max_idx, y_axis_res*50), 0.01, 3, linewidth=3, edgecolor='y', facecolor='none')
    ax.add_patch(rect)
    plt.show()
    return mfccs

def cal_cos(fricative, window_select):
    # fricative width should equal to window_select
    # fricative is (width, n_mfcc)
    # window_select  is (width, n_mfcc)
    # print(fricative.shape, window_select.shape)
    sim = cosine_similarity(fricative, window_select, dense_output=True)
    # sim size would be 1*1
    # test width=1 first
    return sim


def get_fri_indics(Xdb):
    """
    TODO: select proper width for every person, which is the reason to use high frequency microphone
    :param Xdb: STFT result
    :return: indics ranges which indicate fricative or plosive consonant.
    """
    max = 0
    max_idx = 0
    for idx, col in enumerate(Xdb.T):
        high_sum = sum(col[50:600])  # convert this value to MFCC coordinates
        if(high_sum > max):
            max = high_sum
            max_idx = idx
    return max_idx

def get_fri_width(Xdb, max_idx):
    # similarity function to compare look forward and look backward from max_idx
    for idx, col in enumerate(Xdb.T):
        pass




if __name__ == '__main__':
    sr, samples = readfile('jianzhi-hig/2.wav')
    Xdb = get_stft(sr=sr, samples=samples)
    print(Xdb.shape)
    max_idx = get_fri_indics(Xdb)
    mel_basis = gen_mel(48000, 2048, 10)
    get_mfcc(mel_basis, max_idx, Xdb, 192000, 10, log=True)
    print(max_idx)
