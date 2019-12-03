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
from functools import reduce
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity


# sample_rate = 48000
# Given .wav file, extract MFCC features with default
# Test Sliding Windows
def extractMFCC(filename, sample_rate, mels, nmfccs):
    #sample_rate = 192000
    utter_part, sr = librosa.core.load(filename, sample_rate)  # load utterance audio
    intervals = librosa.effects.split(utter_part, top_db=20)  # voice activity detection
    # print(intervals)
    S_total = []
    for inte in intervals:
        S = librosa.core.stft(y=utter_part[inte[0]:inte[1]])
        S = np.abs(S) ** 2
        # print("Size of S is {}".format(S.shape))
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=2048, n_mels=40)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)
        # print(S.shape)
        S_total.append(S)

    # S = librosa.core.stft(y=utter_part)
    # S = np.abs(S) ** 2
    # # print("Size of S is {}".format(S.shape))
    # mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=2048, n_mels=mels)
    # S = np.log10(np.dot(mel_basis, S) + 1e-6)

    # plt.show()
    # plt.pause(2)
    # Extract MFCC feature
    # mfccs = librosa.feature.mfcc(y=utter_part.astype('float'), sr=sample_rate, n_mfcc=10)
    S_total = np.concatenate(S_total,axis=1)
    # print(S_total.shape)
    mfccs = librosa.feature.mfcc(S = S_total, sr=sample_rate, n_mfcc=nmfccs) # (128, times)
    # print(mfccs.shape)
    # print("Shape of mfcc features is {}".format(mfccs.shape))
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfccs, x_axis='time', hop_length=58)
    # plt.colorbar()
    # plt.title('MFCC')
    # plt.tight_layout()
    mfccs = mfccs.reshape(-1,nmfccs)
    # print("Shape of mfcc features is {}".format(mfccs.shape))
    return mfccs

def choose_dataset(dataset):

    nPerson = len(dataset)
    person_choose = np.random.randint(nPerson)
    select_person = os.path.join('.', dataset[person_choose])
    wavs = os.listdir(select_person)
    wav_choose = np.random.randint(len(wavs))
    wav_path = os.path.join(select_person, wavs[wav_choose])
    wav_label = select_person[:-4]
    contain_fricative = False
    if(wav_choose in [1,2,4,5]):
        contain_fricative = True
    return wav_path, wav_label, contain_fricative

# Random select 2 as enrollment, to train GMM model
def split(speaker_wavs):
    # random.shuffle(speaker_wavs)
    # print(speaker_wavs)
    enroll = speaker_wavs[1:3]+speaker_wavs[4:5]+speaker_wavs[5:6]
    verify = speaker_wavs[0:1]+speaker_wavs[3:4]
    return enroll, verify

def speakers_mfccs(dataset):
    speakers_en = {}
    speakers_ve = {}
    for idx, speaker in enumerate(dataset):
        speaker_wavs = os.listdir(os.path.join('.', dataset[idx])) # Get all wavs from the speaker
        speaker_label = dataset[idx][:-4]
        enroll, verify = split(speaker_wavs)
        en_mfccs = []
        ve_mfccs = []
        for en in enroll:
            wav_path = os.path.join('.', dataset[idx], en)
            mfcc = extractMFCC(wav_path, sample_rate=48000, mels=40, nmfccs=40)
            mfcc = mfcc - np.mean(mfcc, axis=0)
            # print(mfcc.shape)
            en_mfccs.append(mfcc)
        speakers_en[speaker_label] = np.concatenate(en_mfccs, axis=0)
        for ve in verify:
            wav_path = os.path.join('.', dataset[idx], ve)
            mfcc = extractMFCC(wav_path, sample_rate=48000, mels=40, nmfccs=40)
            mfcc = mfcc - np.mean(mfcc, axis=0)
            # print(mfcc.shape)
            ve_mfccs.append(mfcc)
        speakers_ve[speaker_label] = np.concatenate(ve_mfccs, axis=0)
    return speakers_en, speakers_ve

def GMMUBMModel(speakers_en):
    GMM = {}
    UBM = {}
    for k, v in speakers_en.items():
        GMM[k] = GaussianMixture(n_components=4, covariance_type='diag')
        UBM[k] = GaussianMixture(n_components=5, covariance_type='diag')
        #print(v.shape)
        GMM[k].fit(v)
        other_v = []
        for k1, v1 in speakers_en.items():
            if (k1 != k):
                other_v.append(v1)
        other_v = np.concatenate(other_v)
        UBM[k].fit(other_v)

    return GMM, UBM

def testGMMUBM(speakers_ve, GMM, UBM, result):
    for k, values in speakers_ve.items():
        print("For speaker {}".format(k))
        # print(GMM[k].score_samples(v))
        # print(speakers_ve.keys())
        # values = speakers_ve['liuli']
        x = GMM[k].score_samples(values) - UBM[k].score_samples(values)
        # print(x.shape)
        total = 0
        correct = 0
        for i in x:
            if i > 0:
                correct += 1
            total += 1
        # Accuracy for every phoneme
        if k not in result.keys():
            result[k] = {'accs':[]}
        result[k]['accs'].append(correct/total)
        #  print("accuracy is {}".format(correct / total))
    return result

def testGMMUBM_utterance(speakers_ve, GMM, UBM, result):
    for k, values in speakers_ve.items():
        print("For speaker {}".format(k))
        # print(GMM[k].score_samples(v))
        # print(speakers_ve.keys())
        err_accept = 0
        err_rej = 0
        for test_person in speakers_ve.keys():
            values = speakers_ve[test_person]
            x = GMM[k].score_samples(values) - UBM[k].score_samples(values)
            # print(x.shape)
            total = 0
            correct = 0
            for i in x:
                if i > 0:
                    correct += 1
                total += 1
            acc = correct / total
            if test_person != k:
                if(acc > 0.3):
                    err_accept = err_accept + 1
            else:
                if(acc < 0.3):
                    err_rej = err_rej + 1
        err_accept_ratio = err_accept / 25.0
        err_rej_ratio = err_rej / 5.0

        # values = speakers_ve['liuli']

        # Accuracy for every phoneme
        if k not in result.keys():
            result[k] = {'err_accept':[], 'err_rej':[]}
        result[k]['err_accept'].append(err_accept_ratio)
        result[k]['err_rej'].append(err_rej_ratio)
        #  print("accuracy is {}".format(correct / total))
    return result

def slidingWindow():
    pass

def testSlidingWindow(fricative):
    fricative = fricative.reshape(-1, 200)
    sumv = 0
    for i in range(round(752 / 20) - 1):
        selected = mfcc3[i * 20:(i + 1) * 20, :]
        selected = selected.reshape(-1, 200)
        # print(selected.shape)
        sim = cosine_similarity(fricative, selected, dense_output=True)
        sumv = sumv + sim[0][0]
        if (sim[0][0] > 0.95):
            print(i, i * 20, sim)


if __name__ == '__main__':
    high_dataset = [dirs for dirs in os.listdir('.') if(dirs[-3:]==("hig"))]
    low_dataset = [dirs for dirs in os.listdir('.') if(dirs[-3:]==("low"))]
    choose_dataset(dataset=high_dataset)
    result = {}
    for i in range(5):
        print("################################################\n")
        print("Round {}\n".format(i))
        print("################################################\n")
        speakers_en, speakers_ve = speakers_mfccs(low_dataset)
        GMM, UBM = GMMUBMModel(speakers_en)
        result = testGMMUBM(speakers_ve, GMM, UBM, result)
    print(result)
    # for k,v in result.items():
    #     result[k]['err_accept_avg'] = sum(result[k]['err_accept'])/ len(result[k]['err_accept'])
    #     result[k]['err_rej_avg'] = sum(result[k]['err_rej']) / len(result[k]['err_rej'])
    # print(result)
    for k,v in result.items():
        result[k]['avg'] = sum(result[k]['accs'])/ len(result[k]['accs'])
        # result[k]['err_rej_avg'] = sum(result[k]['err_rej']) / len(result[k]['err_rej'])
    print(result)



