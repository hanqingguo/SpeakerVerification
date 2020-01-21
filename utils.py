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
    """
    Give sample rate and amplitude samples associate time, calculate STFT result
    :param sr: sample rate
    :param samples: amplitude samples read from wav file
    :return: STFT result, 1025*752, 1025 determined by default librosa.stft function, where n_fft=2048
    752 is determined by sample rate. 192000 => 752
                                       48000 => 188
    """
    X = librosa.stft(samples.astype('float'))
    Xdb = librosa.amplitude_to_db(np.abs(X))

    return Xdb


def draw_fricative(Xdb, lower_bound, upper_bound):

    axes1 = librosa.display.specshow(Xdb, sr=sr, x_axis='frames', y_axis='hz')

    # frequency range from 0 to 1025 bins, because Xdb.shape = (1025, 752)
    # choose frequency range to select fricative consonant

    max_idx = get_fri_indics(Xdb, lower_bound, upper_bound)
    # x_axis_res = 17.35/Xdb.shape[0]
    # y_axis_res = 10/Xdb.shape[1]
    # rect = patches.Rectangle((3.5, 1.5), 0.01, 3, linewidth=3, edgecolor='y', facecolor='none')
    max_xaxis = 752
    max_yaxis = 9.6e4
    y_axis_res = max_yaxis / Xdb.shape[0]
    rect = patches.Rectangle((max_idx, lower_bound * y_axis_res), 0.01, (upper_bound-lower_bound)*y_axis_res,
                             linewidth=2, edgecolor='y', facecolor='none')
    # 坐标是左下角坐标
    axes1.add_patch(rect)
    plt.title("STFT Result")
    plt.show()


def gen_mel(sr, nfft, n_mels):
    # sample rate 太高导致mel filters 在高频处没有响应
    mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=n_mels)
    # print("Shape of Mel Filters : {}".format(mel_basis.shape))
    librosa.display.specshow(mel_basis, x_axis='linear', sr=sr)
    plt.title("Mel Filters")
    plt.show()
    print("Mel_filters shape is : {}".format(mel_basis.shape))
    return mel_basis


def get_mfcc(mel_basis, max_idx, Xdb, sr, n_mfcc, log=True):
    dot_result = np.dot(mel_basis, Xdb)
    if (log):
        dot_result = np.log10(dot_result + 1e-6)
    librosa.display.specshow(dot_result, sr=sr, x_axis='time', y_axis='hz')
    plt.title("Result of STFT dot multiply Mel-filters")
    plt.show()

    mfccs = librosa.feature.mfcc(S=dot_result, sr=sr, n_mfcc=n_mfcc)
    fig, ax = plt.subplots(1)
    librosa.display.specshow(mfccs, x_axis='time')
    x_axis_res = 17.35 / Xdb.shape[1]
    y_axis_res = 10 / Xdb.shape[0]
    print(x_axis_res * max_idx, y_axis_res * 50)
    rect = patches.Rectangle((x_axis_res * max_idx, y_axis_res * 50), 0.01, 3, linewidth=3, edgecolor='y',
                             facecolor='none')
    ax.add_patch(rect)
    plt.title("MFCC result")
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


def get_fri_indics(Xdb, lower_bound, upper_bound):
    """
    TODO: select proper width for every person, which is the reason to use high frequency microphone
    :param Xdb: STFT result
    :return: indics ranges which indicate fricative or plosive consonant.
    """
    max = 0
    max_idx = 0
    # freq_range = {"lower": lower_bound, "upper" : upper_bound}
    for idx, col in enumerate(Xdb.T):
        high_sum = sum(col[lower_bound: upper_bound])  # convert this value to MFCC coordinates
        if (high_sum > max):
            max = high_sum
            max_idx = idx
    return max_idx


def get_fri_width(mfccs, max_idx):
    # similarity function to compare look forward and look backward from max_idx
    """
    Given mfcc feature and fricative index, find the fricative consonent width
    :param mfccs: MFCC feature
    :param max_idx: fricative consonant index found by function get_fri_indics
    :return: width of fricative consonant
    """
    print(mfccs.shape)
    # mfcc.shape = (10, 752) 降维提升运算速度
    fricative_componant = mfccs[:, max_idx]
    fricative_componant = fricative_componant.reshape(-1, mfccs.shape[0])
    print(mfccs[:, max_idx:].shape)

    # im = cosine_similarity(fricative_componant, window_select, dense_output=True)s

    for idx, col in enumerate(mfccs[:, max_idx:].T):
        col = col.reshape(-1, mfccs.shape[0])
        sim = cosine_similarity(fricative_componant, col, dense_output=True)
        print(sim)

    return width


if __name__ == '__main__':
    sr, samples = readfile('hanqing-hig/5.wav')
    lower_bound = 50
    upper_bound = 800
    Xdb = get_stft(sr=sr, samples=samples)
    draw_fricative(Xdb, lower_bound, upper_bound)
    print(Xdb.shape)
    max_idx = get_fri_indics(Xdb, lower_bound, upper_bound)
    mel_basis = gen_mel(48000, 2048, 10)
    mfcc = get_mfcc(mel_basis, max_idx, Xdb, 192000, 10, log=True)
    width = get_fri_width(mfcc, max_idx)
