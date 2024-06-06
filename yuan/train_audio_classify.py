# -*- coding: utf-8 -*-
import sys

import scipy

sys.path.append('../../..')

import warnings

warnings.filterwarnings("ignore")
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from glob import glob
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# 模型结构
class ClassifyModel:
    def __init__(self, feature_num, out_num):
        self.output_num = out_num
        self.feature_num = feature_num
        self.input_feature = layers.Input(shape=[None, feature_num], name='input_feature')

        # 模型输入输出
        model_output = self.generate_model()
        self.model = tf.keras.Model(inputs=self.input_feature, outputs=model_output)
        self.model.summary()

    def generate_model(self):  # 构建模型
        layer_output = self.input_feature[:, :, :, tf.newaxis]

        # conve2d卷积 bn层
        layer_output = layers.Conv2D(48, kernel_size=[3, 3], padding='valid')(layer_output)
        layer_output = layers.BatchNormalization()(layer_output)
        layer_output = layers.ReLU()(layer_output)
        layer_output = layers.Dropout(0.05)(layer_output)

        layer_output = layers.Conv2D(48, kernel_size=[3, 3], padding='valid')(layer_output)
        layer_output = layers.BatchNormalization()(layer_output)
        layer_output = layers.ReLU()(layer_output)
        layer_output = layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(layer_output)
        layer_output = layers.Dropout(0.15)(layer_output)

        layer_output = layers.Conv2D(72, kernel_size=[3, 3], padding='valid')(layer_output)
        layer_output = layers.BatchNormalization()(layer_output)
        layer_output = layers.ReLU()(layer_output)
        layer_output = layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(layer_output)
        layer_output = layers.Dropout(0.25)(layer_output)

        layer_output = layers.Reshape((-1, layer_output.shape[2] * layer_output.shape[3]), name='Reshape0')(
            layer_output)  # Reshape层
        layer_output = layers.Dense(512)(layer_output)  # 全连接层
        layer_output = layers.ReLU()(layer_output)

        layer_output = layers.Bidirectional(layers.LSTM(256, return_sequences=False))(layer_output)
        layer_output = layers.Flatten()(layer_output)
        layer_output = layers.Dense(self.output_num)(layer_output)
        layer_output = layers.Softmax(-1)(layer_output)
        # layer_output = tf.nn.sigmoid(layer_output) # 二分类sigmoid就可以

        return layer_output

    def res2net_plus_edit(self):
        slice_num = 8
        num_filters = 512

        x = self.input_feature[:, :, :, tf.newaxis]
        x_conv_nor = self.conv_bn_relu(num_filters, (3, 3))(x)
        slice_list = self.slice_layer(x_conv_nor, slice_num, num_filters)

        side = self.conv_bn_relu_2(num_filters=num_filters // slice_num, kernel_size=(3, 3))(slice_list[1])
        z = tf.concat([slice_list[0], side], -1)  # for one and second stage
        for i in range(2, len(slice_list)):
            y = self.conv_bn_relu_2(num_filters=num_filters // slice_num, kernel_size=(3, 3))(side + slice_list[i])
            side = y
            z = tf.concat([z, y], -1)
        z = x_conv_nor + z
        z = self.conv_bn_relu_2(num_filters=64, kernel_size=(1, 1))(z)
        z = layers.Reshape([-1, z.shape[2] * z.shape[3]])(z)
        z = layers.Dense(256)(z)
        z = self.mish(z)

        x = z
        att_memory = []
        for i in range(6):
            residual = x
            x1 = layers.Dense(units=256, kernel_initializer='he_normal')(x)
            x2 = layers.Dense(units=256, kernel_initializer='he_normal')(x)
            x = x1 * tf.nn.sigmoid(x2)
            x = layers.Dropout(0.1)(x)
            x = tf.multiply(0.5, x)
            x = layers.Add()([residual, x])
            x = layers.LayerNormalization(epsilon=1e-5)(x)

            # encoder-self-attention
            residual = x
            if len(att_memory) > 0:
                x = x + att_memory[-1]
            x = layers.Attention(256)([x, x, x])
            att_memory.append(x)
            x = layers.Dense(units=256, kernel_initializer='he_normal')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Add()([residual, x])
            x = layers.LayerNormalization(epsilon=1e-5)(x)

            # conv_module:conformer对比transformer最大的改变
            # LN-point-GLU
            residual = x
            encoder_output_point1 = layers.Conv1D(filters=256, kernel_size=1, strides=1)(x)  # pointwise
            encoder_output_point2 = layers.Conv1D(filters=256, kernel_size=1, strides=1)(x)  # pointwise
            x = encoder_output_point1 * tf.sigmoid(encoder_output_point2)  # GLU

            # 分离卷积+因果卷积+BN+swish
            x = layers.Conv1D(filters=256, kernel_size=15, strides=1, padding='causal', dilation_rate=2,
                              groups=x.shape[-1])(x)  # depthwise-dilate
            x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
            x = self.swish(x)
            # point-residual
            x = layers.Conv1D(filters=256, kernel_size=1, strides=1)(x)  # pointwise
            x = layers.Dropout(0.1)(x)
            x = layers.Add()([residual, x])
            x = layers.LayerNormalization(epsilon=1e-5)(x)

            # encoder-feed-forward
            residual = x
            x1 = layers.Dense(units=256, kernel_initializer='he_normal')(x)
            x2 = layers.Dense(units=256, kernel_initializer='he_normal')(x)
            x = x1 * tf.nn.sigmoid(x2)
            x = layers.Dropout(0.1)(x)
            x = tf.multiply((1 - 0.5), x)
            x = layers.Add()([residual, x])
            x = layers.LayerNormalization(epsilon=1e-5)(x)

        z = layers.Dense(1)(x)
        z = tf.squeeze(z, -1)
        z = tf.sigmoid(z)
        return z

    def mish(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

    def conv_bn_relu(self, num_filters,
                     kernel_size,
                     strides=(1, 1),
                     padding='same', ):
        def layer(input_tensor):
            x = layers.Conv2D(num_filters, kernel_size,
                              padding=padding, kernel_initializer='he_normal',
                              strides=strides)(input_tensor)
            x = self.mish(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            return x

        return layer

    def slice_layer(self, x, slice_num, channel_input):
        output_list = []
        single_channel = channel_input // slice_num
        for i in range(slice_num):
            out = x[:, :, :, i * single_channel:(i + 1) * single_channel]
            output_list.append(out)
        return output_list


# 输出处理类
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.audio_max_len = 128

    # 获取输入输出
    def __getitem__(self, index):
        wav_path = self.data[index][0]
        label = self.data[index][1]

        # 提取特征
        audio_feature = self.extract_feature(wav_path)
        # 随机选取一个片段

        start_id = len(audio_feature) - self.audio_max_len
        if start_id < 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_id)

        end_id = start_id + self.audio_max_len
        audio_feature = audio_feature[start_id:end_id]
        audio_feature = self.pad_zero(audio_feature, self.audio_max_len)

        return (audio_feature, label)

    # 提取音频logfbank特征
    def extract_feature(self, wav_path):
        win_length = 400
        n_fft = 512
        hop_length = 160
        sr = 16000
        # 音频时域信息
        audio_samples = librosa.load(wav_path, sr=sr)[0]
        # 音频的mel频谱信息
        mel = librosa.feature.melspectrogram(audio_samples, sr, hop_length=hop_length, win_length=win_length,
                                             n_fft=n_fft, n_mels=feature_dim).astype(np.float32).T
        # 将mel信息转为对应的db信息
        mel = librosa.power_to_db(mel)
        mfcc = scipy.fftpack.dct(mel, axis=0, type=2, norm='ortho')[:80]
        return mfcc

    # 进行补零
    def pad_zero(self, input, length):
        input_shape = input.shape
        if input_shape[0] >= length:
            return input[:length]

        if len(input_shape) == 1:
            return np.append(input, [0] * (length - input_shape[0]), axis=0)

        if len(input_shape) == 2:
            return np.append(input, [[0] * input_shape[1]] * (length - input_shape[0]), axis=0)

    def __len__(self):
        return len(self.data)


# 准备数据
def pre_data():
    label = ['about', 'beat', 'bed', 'bird', 'bit', 'boot', 'bought', 'but', 'cat', 'father', 'pot', 'put']
    audio_files = glob('/data/nas/aim/en_yuanyin/mu/*.wav')

    train_data = []
    for audio_file in audio_files:
        file_name = os.path.splitext(os.path.split(audio_file)[1])[0].split('_')[0]
        train_data.append([audio_file, label.index(file_name)])

    audio_files = glob('/data/nas/aim/en_yuanyin/xue/*.wav')

    test_data = []
    for audio_file in audio_files:
        file_name = os.path.splitext(os.path.split(audio_file)[1])[0].split('_')[0]
        test_data.append([audio_file, label.index(file_name)])
    return train_data, test_data, label


# 计算损失函数
def compute_loss(label, pre):
    # 交叉熵损失函数
    loss = -tf.multiply(label, tf.keras.backend.log(pre + 1e-7)) - tf.multiply(
        (1 - label), tf.keras.backend.log(1 - pre + 1e-7))
    loss = tf.reduce_mean(loss)
    return loss


def load_frame(path, frame_name='frame'):  # 加载frame文件
    frame_names = {}
    for frame_name in glob(f'{path}/{frame_name}*'):
        name = os.path.split(frame_name)[1]
        frame_names[int(name.split('_')[-1])] = frame_name

    if len(sorted(frame_names)) == 0:
        return None, None
    else:
        frame_index = sorted(frame_names)[-1]
        return frame_names[frame_index], frame_index


def delete_frame(path, frame_name='frame'):  # 删除frame文件
    frame_names = {}
    for frame_name in glob(f'{path}/{frame_name}*'):
        name = os.path.split(frame_name)[1]
        frame_names[int(name.split('_')[-1])] = frame_name

    for delete_key in sorted(frame_names)[:-5]:
        os.remove(frame_names[delete_key])


if __name__ == '__main__':
    # 参数
    batch_size = 8
    num_epoch = 20
    process_num = 3
    lr = 0.0006
    feature_dim = 80
    pb_path = f'resources/classify_model'
    os.makedirs(pb_path, exist_ok=True)

    ''' ------------------------准备数据------------------------------ '''
    train_data, test_data, label = pre_data()
    out_num = len(label)

    ''' ------------------------加载模型------------------------------ '''
    classify_model = ClassifyModel(feature_dim, out_num).model

    # 恢复权重
    frame_path, frame_index = load_frame(path=pb_path)
    if frame_path is not None:
        classify_model.load_weights(frame_path)
        print(f'恢复权重 = {classify_model}')

    ''' ------------------------开始训练------------------------------ '''
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                     decay_steps=10000,
                                                                     decay_rate=0.98))

    train_loader = DataLoader(dataset=AudioDataset(train_data), batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=5, collate_fn=None)
    history = {'train_acc': [], 'train_recall': [], 'train_f1': [], 'train_loss': [], }

    for epoch_index in range(1, num_epoch):
        acc, f1, recall, count = 0, 0, 0, 0
        train_tqdm = tqdm(train_loader)
        for input_feature, input_label in train_tqdm:
            input_feature = input_feature.numpy()
            input_label = input_label.numpy()

            # 梯度下降
            with tf.GradientTape(persistent=False) as tape:
                output = classify_model(input_feature, training=True)
                loss = compute_loss(tf.one_hot(input_label, depth=out_num), output)

                grads = tape.gradient(target=loss, sources=classify_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, classify_model.trainable_variables))  # step自动加一

                output_true = np.argmax(output, -1)
                # 宏观统计所有，微观分别统计再平均，权重分别统计根据数量权重平均
                acc += accuracy_score(input_label, output_true)
                f1 += f1_score(input_label, output_true, average='weighted')
                recall += recall_score(input_label, output_true, average='weighted')
                count += 1
                train_tqdm.set_description(f'epoch = {epoch_index}  loss = {loss}')
                history['train_loss'].append(loss)

        recall = recall / count
        f1 = f1 / count
        acc = acc / count

        history['train_acc'].append(acc)
        history['train_f1'].append(f1)
        history['train_recall'].append(recall)

        classify_model.save(f'{pb_path}/frame_{epoch_index}', save_format='h5')
        classify_model.save(f'{pb_path}/pb/frame.h5', save_format='tf')
        delete_frame(pb_path)

    plt.figure(figsize=(9, 5))
    x = np.arange(1, len(history['train_loss']) + 1)
    plt.plot(x, history['train_loss'], label='train_loss')
    plt.xlabel('step')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(9, 5))
    x = np.arange(1, len(history['train_recall']) + 1)
    plt.plot(x, history['train_recall'], label='train_recall')
    plt.plot(x, history['train_f1'], label='train_f1')
    plt.plot(x, history['train_acc'], label='train_acc')
    plt.xlabel('Epoch')
    plt.title('Metrics')
    plt.legend()
    plt.show()

    print(f'训练完成')
