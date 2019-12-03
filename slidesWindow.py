from utils import *
import librosa


def get_features(file):
    sr, samples = readfile(file)
    Xdb = get_stft(sr=sr, samples=samples)
    mel_basis = gen_mel(44000, nfft=2048, n_mels=10)
    mfccs = get_mfcc(mel_basis, Xdb, sr, n_mfcc=10, log=False)
    idx = get_fri_indics(Xdb)
    #print("High_sum is {}".format(idx))
    return mfccs, idx


def verify(speaker, wavs, threshold):
    """
    Given one speaker, and test all speakers wav files
    :param speaker: the enrolled speaker
    :param wavs: all wav files
    :param threshold:
    :return: speaker, {threshold:[], FAR:[], FRR:[], EER}
    """
    speaker_wavs = os.listdir(speaker)
    fricative = os.path.join(speaker, '2.wav')
    mfcc, idx = get_features(fricative)
    model = mfcc[1:, idx].reshape(1,-1)
    fail_accept = 0
    fail_reject = 0

    for wav in wavs:
        test_mfcc, test_idx = get_features(wav)
        sim = cosine_similarity(model, test_mfcc[1:, test_idx].reshape(1, -1))
        print("Speaker Model is {}, test voice from : {}\n, sim : {}\n".format(speaker, wav[:-6], sim))
        if(speaker == wav[:-6]):
            if(sim < threshold):
                fail_reject += 1
        else:
            if(sim > threshold):
                fail_accept += 1
    print("Speaker Model is {}, threshold : {}\n, FAR : {}/12, FRR : {}/6\n".format(speaker, threshold, fail_accept, fail_reject))
    return threshold, fail_accept*100/12, fail_reject*100/6


if __name__ == '__main__':

    wavs = []
    dataset = [dirs for dirs in os.listdir('.') if (dirs[-4:] == ("-hig"))]
    print(dataset)
    for speaker in dataset:
        for wav in os.listdir(speaker):
            wavs.append(os.path.join(speaker, wav))

    # thes = []
    # FARs = []
    # FRRs = []
    # for threshold in np.arange(0.5, 1.0, 0.025):
    #     the, FAR, FRR = verify('Nick-hig', wavs, threshold)
    #     thes.append(the)
    #     FARs.append(FAR)
    #     FRRs.append(FRR)
    # plt.plot(thes, FARs, marker='o', label = "FAR", markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    # plt.plot(thes, FRRs, marker='', label = "FRR", color='olive', linewidth=2)


    for idx, speaker in enumerate(dataset):
        thes = []
        FARs = []
        FRRs = []
        for threshold in np.arange(0.5,1.0,0.025):
            the, FAR, FRR = verify(speaker, wavs, threshold)
            thes.append(the)
            FARs.append(FAR)
            FRRs.append(FRR)
        FARs = np.array(FARs)
        FRRs = np.array(FRRs)
        if idx == 0:
            FARss = FARs
            FRRss = FRRs
        else:
            FARss = FARss + FARs
            FRRss = FRRss + FRRs
    


    plt.plot(thes, FARss/3, marker='o', label = "FAR", markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    plt.plot(thes, FRRss/3, marker='', label = "FRR", color='olive', linewidth=2)


    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("%")
    plt.tick_params(labelright=True)
    plt.title('All')
    plt.show()
