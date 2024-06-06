import librosa.display
import scipy
from tqdm import tqdm
import tensorflow as tf

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from glob import glob
import os
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '2'




def lpc_coeff(s, p):
    """
    :param s: 一帧数据
    :param p: 线性预测的阶数
    :return:
    """
    n = len(s)
    # 计算自相关函数
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
    Rp0 = np.matmul(s, s.T)
    Ep = np.zeros((p, 1))
    k = np.zeros((p, 1))
    a = np.zeros((p, p))
    # 处理i=0的情况
    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0, 0] = k[0]
    Ep[0] = (1 - k[0] * k[0]) * Ep0
    # i=1开始，递归计算
    if p > 1:
        for i in range(1, p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
            a[i, i] = k[i]
            Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
            for j in range(i - 1, -1, -1):
                a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
    ar = np.zeros(p + 1)
    ar[0] = 1
    ar[1:] = -a[:, p - 1]
    G = np.sqrt(Ep[p - 1])
    return ar, G


def local_maxium(x):
    """
    求序列的极大值
    :param x:
    :return:
    """
    d = np.diff(x)
    l_d = len(d)
    maxium = []
    loc = []
    for i in range(l_d - 1):
        if d[i] > 0 and d[i + 1] <= 0:
            maxium.append(x[i + 1])
            loc.append(i + 1)
    return maxium, loc


def Formant_Cepst(u, cepstL):
    """
    倒谱法共振峰估计函数
    :param u:输入信号
    :param cepstL:🔪频率上窗函数的宽度
    :return: val共振峰幅值 loc共振峰位置 spec包络线
    """
    wlen2 = len(u) // 2
    u_fft = np.fft.fft(u)  # 按式（2-1）计算
    U = np.log(np.abs(u_fft[:wlen2]))
    Cepst = np.fft.ifft(U)  # 按式（2-2）计算
    cepst = np.zeros(wlen2, dtype=np.complex)
    cepst[:cepstL] = Cepst[:cepstL]  # 按式（2-3）计算
    cepst[-cepstL + 1:] = Cepst[-cepstL + 1:]  # 取第二个式子的相反
    spec = np.real(np.fft.fft(cepst))
    val, loc = local_maxium(spec)  # 在包络线上寻找极大值
    return val, loc, spec


def Formant_Interpolation(u, p, fs):
    """
    插值法估计共振峰函数
    :param u:
    :param p:
    :param fs:
    :return:
    """
    ar, _ = lpc_coeff(u, p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    df = fs / 512
    val, loc = local_maxium(U)
    ll = len(loc)
    pp = np.zeros(ll)
    F = np.zeros(ll)
    Bw = np.zeros(ll)
    for k in range(ll):
        m = loc[k]
        m1, m2 = m - 1, m + 1
        p = val[k]
        p1, p2 = U[m1], U[m2]
        aa = (p1 + p2) / 2 - p
        bb = (p2 - p1) / 2
        cc = p
        dm = -bb / 2 / aa
        pp[k] = -bb * bb / 4 / aa + cc
        m_new = m + dm
        bf = -np.sqrt(bb * bb - 4 * aa * (cc - pp[k] / 2)) / aa
        F[k] = (m_new - 1) * df
        Bw[k] = bf * df
    return F, Bw, pp, U, loc


def Formant_Root(u, p, fs, n_frmnt):
    """
    LPC求根法的共振峰估计函数
    :param u:
    :param p:
    :param fs:
    :param n_frmnt:
    :return:
    """
    ar, _ = lpc_coeff(u, p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    const = fs / (2 * np.pi)
    rts = np.roots(ar)
    yf = []
    Bw = []
    for i in range(len(ar) - 1):
        re = np.real(rts[i])
        im = np.imag(rts[i])
        fromn = const * np.arctan2(im, re)
        bw = -2 * const * np.log(np.abs(rts[i]))
        if fromn > 150 and bw < 700 and fromn < fs / 2:
            yf.append(fromn)
            Bw.append(bw)
    return yf[:min(len(yf), n_frmnt)], Bw[:min(len(Bw), n_frmnt)], U


def plt_info():
    audio_path = 'F:/thesis/mu/about_1.wav'
    plt.figure(figsize=(20, 18))

    data = librosa.load(audio_path, 16000)[0]
    fs = 16000
    # 预处理-预加重
    u = lfilter([1, -0.99], [1], data)

    cepstL = 6
    wlen = len(u)
    wlen2 = wlen // 2
    # 预处理-加窗
    u2 = np.multiply(u, np.hamming(wlen))
    # 预处理-FFT,取对数
    U_abs = np.log(np.abs(np.fft.fft(u2))[:wlen2])
    freq = [i * fs / wlen for i in range(wlen2)]
    val, loc, spec = Formant_Cepst(u, cepstL)
    plt.subplot(4, 1, 1)
    plt.plot(freq, U_abs, 'k')
    plt.title('Spectrum')
    plt.subplot(4, 1, 2)
    plt.plot(freq, spec, 'k')
    plt.title('Cepstrum method for formant estimation')
    for i in range(len(loc)):
        plt.subplot(4, 1, 2)
        plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(spec), spec[loc[i]]], '-.k')
        plt.text(freq[loc[i]], spec[loc[i]], 'Freq={}'.format(int(freq[loc[i]])))
    p = 12
    freq = [i * fs / 512 for i in range(256)]
    F, Bw, pp, U, loc = Formant_Interpolation(u, p, fs)

    plt.subplot(4, 1, 3)
    plt.plot(freq, U)
    plt.title('Formant Estimation Using LPC Interpolation')

    for i in range(len(Bw)):
        plt.subplot(4, 1, 3)
        plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(U), U[loc[i]]], '-.k')
        plt.text(freq[loc[i]], U[loc[i]], 'Freq={:.0f}\nHp={:.2f}\nBw={:.2f}'.format(F[i], pp[i], Bw[i]))

    p = 12
    freq = [i * fs / 512 for i in range(256)]

    n_frmnt = 4
    F, Bw, U = Formant_Root(u, p, fs, n_frmnt)

    plt.subplot(4, 1, 4)
    plt.plot(freq, U)
    plt.title('Formant Estimation Using LPC Root-finding Method')

    for i in range(len(Bw)):
        plt.subplot(4, 1, 4)
        plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(U), U[loc[i]]], '-.k')
        plt.text(freq[loc[i]], U[loc[i]], 'Freq={:.0f}\nBw={:.2f}'.format(F[i], Bw[i]))

    # plt.savefig('images/共振峰估计.png')
    plt.show()
    plt.close()
    exit()

    # 加载音频文件
    audio_path = 'F:/thesis/mu/about_1.wav'
    y, sr = librosa.load(audio_path, sr=16000)
    # 计算线性预测系数（LPC）
    lpc_order = 2 + int(sr / 1000)  # 通常取采样率的2%作为LPC阶数
    r = librosa.lpc(y, lpc_order)

    # 计算多项式的根，即共振峰频率
    roots = np.roots(r)

    # 计算共振峰频率（单位：Hz）
    frequencies = np.abs(roots * (2 * np.pi / sr))

    # 过滤掉非共振峰频率
    # 假设人声共振峰频率范围是300Hz到3400Hz
    formants = frequencies[(frequencies > 300) & (frequencies < 3400)]

    # 可视化共振峰
    plt.figure(figsize=(10, 4))
    plt.plot(formants)
    plt.title('Resonance Frequencies')
    plt.xlabel('Formant Index')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def pre_data():
    audio_files = glob('F:/thesis/mu/*.wav')

    data = []
    for audio_file in audio_files:
        file_name = os.path.splitext(os.path.split(audio_file)[1])[0].split('_')[0]
        data.append([audio_file, file_name])

    label = sorted(list(set([d[1] for d in data])))
    data = [[d[0], label.index(d[1])] for d in data]

    train_data = data
    test_data = glob('F:/thesis/xue/*.wav')
    return train_data, test_data, label


def get_formant(audio_file):
    data = librosa.load(audio_file, 16000)[0]
    fs = 16000
    # 预处理-预加重
    u = lfilter([1, -0.99], [1], data)

    cepstL = 6
    wlen = len(u)
    wlen2 = wlen // 2
    # 预处理-加窗
    u2 = np.multiply(u, np.hamming(wlen))
    # 预处理-FFT,取对数
    U_abs = np.log(np.abs(np.fft.fft(u2))[:wlen2])
    freq = [i * fs / wlen for i in range(wlen2)]
    val, loc, spec = Formant_Cepst(u, cepstL)
    return val, loc, spec


def extract_feature(wav_path):
    win_length = 400
    n_fft = 512
    hop_length = 160
    sr = 16000
    # 音频时域信息
    audio_samples = librosa.load(wav_path, sr=sr)[0]
    # 音频的mel频谱信息
    mel = librosa.feature.melspectrogram(audio_samples, sr, hop_length=hop_length, win_length=win_length,
                                         n_fft=n_fft, n_mels=80).astype(np.float32).T
    # 将mel信息转为对应的db信息
    mel = librosa.power_to_db(mel)
    mfcc = scipy.fftpack.dct(mel, axis=0, type=2, norm='ortho')[:80]
    return mfcc


if __name__ == '__main__':
    model_file = 'F:/thesis/yuan/resources/classify_model/pb/frame.h5'
    infer_model = tf.keras.models.load_model(model_file)



    train_data, test_data, label = pre_data()

    formant_dict = defaultdict(list)
    for data in tqdm(train_data):
        audio_file = data[0]
        file_label = label[data[1]]
        audio_formant = get_formant(audio_file)[1]
        formant_dict[file_label].append(audio_formant[:3])

    for key in formant_dict:
        sub_data = np.array(formant_dict[key])
        sub_data_max = np.max(sub_data, 0)
        sub_data_min = np.min(sub_data, 0)
        formant_dict[key] = np.array([sub_data_min, sub_data_max])

    for audio_file in test_data:
        file_name = os.path.splitext(os.path.split(audio_file)[1])[0].split('_')[0]

        audio_feature = extract_feature(audio_file)
        infer_result = infer_model(audio_feature[np.newaxis, :, :])[0].numpy()

        audio_formant = get_formant(audio_file)[1][:3]
        audio_formatn_label = formant_dict[file_name]

        result = []
        for i in range(3):
            if i == 0:  # 第一共振峰
                if audio_formant[i] < audio_formatn_label[0][i]:
                    result.append('Recommend that the tongue be slightly raised.')
                elif audio_formant[i] < audio_formatn_label[1][i]:
                    pass
                else:
                    result.append('Suggest to lower the tongue slightly.')

            elif i == 1:  # 第二共振峰
                if audio_formant[i] < audio_formatn_label[0][i]:
                    result.append('It is recommended that the tongue be stretched forward.')
                elif audio_formant[i] < audio_formatn_label[1][i]:
                    pass
                else:
                    result.append('It is recommended that the tongue be retracted.')

        print(f"{audio_file}    grade = {infer_result[label.index(file_name)] * 100}   propose = {','.join(result)}")

        # infer_index = np.argmax(infer_result)
        # print(audio_file,label[infer_index])
