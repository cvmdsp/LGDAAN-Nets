import numpy as np
import tensorflow as tf
import scipy.io
import os


def load_data(source_time_path, source_spectral_path, target_time_path, target_spectral_path):
    # 初始化存储数据的列表
    source_time_data = []
    source_spectral_data = []
    target_time_data = []
    target_spectral_data = []
    source_labels = []
    target_labels = []

    # 读取源域和目标域时间数据和光谱数据
    for file_name in os.listdir(source_time_path):
        if file_name.endswith('.mat'):
            # 加载每个.mat文件的数据
            mat_data = scipy.io.loadmat(os.path.join(source_time_path, file_name))
            source_time_data.append(mat_data['data'])
            source_labels.append(mat_data['arousal_labels'])

    for file_name in os.listdir(source_spectral_path):
        if file_name.endswith('.mat'):
            mat_data = scipy.io.loadmat(os.path.join(source_spectral_path, file_name))
            source_spectral_data.append(mat_data['data'])

    for file_name in os.listdir(target_time_path):
        if file_name.endswith('.mat'):
            mat_data = scipy.io.loadmat(os.path.join(target_time_path, file_name))
            target_time_data.append(mat_data['data'])
            target_labels.append(mat_data['arousal_labels'])

    for file_name in os.listdir(target_spectral_path):
        if file_name.endswith('.mat'):
            mat_data = scipy.io.loadmat(os.path.join(target_spectral_path, file_name))
            target_spectral_data.append(mat_data['data'])

    # 将数据转换为 TensorFlow 张量
    source_time_data = tf.convert_to_tensor(np.array(source_time_data), dtype=tf.float32)
    source_spectral_data = tf.convert_to_tensor(np.array(source_spectral_data), dtype=tf.float32)
    target_time_data = tf.convert_to_tensor(np.array(target_time_data), dtype=tf.float32)
    target_spectral_data = tf.convert_to_tensor(np.array(target_spectral_data), dtype=tf.float32)

    # 将标签转换为 TensorFlow 张量
    source_labels = tf.convert_to_tensor(np.array(source_labels), dtype=tf.int32)
    target_labels = tf.convert_to_tensor(np.array(target_labels), dtype=tf.int32)

    return source_time_data, source_spectral_data, target_time_data, target_spectral_data, source_labels, target_labels
