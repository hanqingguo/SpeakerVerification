from PIL import Image
from functools import reduce
from utils import *
from slidesWindow import *
import matplotlib.pyplot as plt

def phash(img):
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    return reduce(
        lambda x, y: x | (y[1] << y[0]),
        enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())),
        0
    )

# 计算汉明距离:
def hamming_distance(a, b):
    return bin(a^b).count('1')

# 计算两个图片是否相似:
def is_imgs_similar(img1,img2):
    print("Hamming_distance is {}".format(hamming_distance(phash(img1),phash(img2))))
    return True if hamming_distance(phash(img1),phash(img2)) <= 5 else False

if __name__ == '__main__':
    img1 = Image.open('hanqing1.png')
    img2 = Image.open('hanqing_clear1.png')
    print(is_imgs_similar(img1=img1, img2=img2))
    # next step: find fricative indics
    # slides width for each block
    sr, samples = readfile('hanqing-hig/2.wav')
    Xdb = get_stft(sr=sr, samples=samples)
    max_idx = get_fri_indics(Xdb)
    mel_basis = gen_mel(48000, 2048, 50)
    get_mfcc(mel_basis, Xdb, 192000, 50, log=True)