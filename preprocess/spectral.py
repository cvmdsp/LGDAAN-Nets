import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    # print(data.shape)
    return data


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def decompose(file):
    # trial*channel*sample
    start_index = 384  # 3s pre-trial signals
    data = read_file(file)
    shape = data.shape
    frequency = 128

    decomposed_de = np.empty([0, 4, 60])
    base_DE = np.empty([0, 128])

    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])

        temp_de = np.empty([0, 60])

        for channel in range(32):
            trial_signal = data[trial, channel, 384:]
            base_signal = data[trial, channel, :384]

            # ****************Perform baseline correction****************
            base_mean = np.mean(base_signal, axis=0)  # Calculate the mean of the baseline period (first 384 samples)
            trial_signal = trial_signal - base_mean  # Subtract the baseline mean from the trial signal

            # ****************Compute base DE****************
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            base_theta_DE = (compute_DE(base_theta[:128]) + compute_DE(base_theta[128:256]) + compute_DE(
                base_theta[256:])) / 3
            base_alpha_DE = (compute_DE(base_alpha[:128]) + compute_DE(base_alpha[128:256]) + compute_DE(
                base_alpha[256:])) / 3
            base_beta_DE = (compute_DE(base_beta[:128]) + compute_DE(base_beta[128:256]) + compute_DE(
                base_beta[256:])) / 3
            base_gamma_DE = (compute_DE(base_gamma[:128]) + compute_DE(base_gamma[128:256]) + compute_DE(
                base_gamma[256:])) / 3

            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE, base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE)

            # ****************Filter the trial signal****************
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(60):
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 128:(index + 1) * 128]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 128:(index + 1) * 128]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 128:(index + 1) * 128]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 128:(index + 1) * 128]))

            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

        # Reshaping the data for the trial
        temp_trial_de = temp_de.reshape(-1, 4, 60)
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

        # Aggregate the base DE features
        temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)
        base_DE = np.vstack([base_DE, temp_base_DE])

    decomposed_de = decomposed_de.reshape(-1, 32, 4, 60).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1,128)

    return base_DE, decomposed_de


def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, 0, data[0], 0, data[16], 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[1], 0, data[17], 0, 0, 0)
    data_2D[2] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[3] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[4] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[5] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[6] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[7] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[8] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    return data_2D
def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[1]):
        norm_dataset_1D[:, i] = feature_normalize(dataset_1D[:, i])
    # return shape: m*32
    return norm_dataset_1D

def get_labels(file):
    # 0 valence, 1 arousal, 2 dominance, 3 liking
    valence_labels = sio.loadmat(file)["labels"][:, 0] > 5  # valence labels
    arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5  # arousal labels
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    for i in range(len(valence_labels)):
        for j in range(0, 60):
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
    print("labels:", final_arousal_labels.shape)
    return final_arousal_labels, final_valence_labels


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized
def get_vector_deviation(vector1,vector2):
	return vector1-vector2 #实现两个向量相减
#计算偏差。。减去基线
def get_dataset_deviation(trial_data,base_data):
	new_dataset = np.empty([0,128])
	for i in range(0,2400):
		base_index = i//60#整除运算，结果取比商小的最大整数
		base_index = 39 if base_index == 40 else base_index
		new_record = get_vector_deviation(trial_data[i],base_data[base_index]).reshape(1,128)
		new_dataset = np.vstack([new_dataset,new_record])
	return new_dataset

def pre_process(trial_data,base_data):
	# DE feature vector dimension of each band
	data_3D = np.empty([0,9,9])
	sub_vector_len = 32
	data = get_dataset_deviation(trial_data,base_data)

	data = preprocessing.scale(data,axis=1, with_mean=True,with_std=True,copy=True)
	# convert 128 vector ---> 4*9*9 cube
	for vector in data:
		for band in range(0,4):
			data_2D_temp = data_1Dto2D(vector[band*sub_vector_len:(band+1)*sub_vector_len])
			data_2D_temp = data_2D_temp.reshape(1,9,9)
			data_3D = np.vstack([data_3D,data_2D_temp])
	data_3D = data_3D.reshape(-1,4,9,9)
	print("final data shape:",data_3D.shape)
	return data_3D


if __name__ == '__main__':
    dataset_dir = "./DEAP_data/"

    result_dir = "./DEAP_spec/"
    if os.path.isdir(result_dir) == False:
        os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        base_DE_n, trial_DE_n = decompose(file_path)

        arousal_labels, valence_labels = get_labels(file_path)

        base_DE_normalized = feature_normalize(base_DE_n)
        trial_DE_normalized = feature_normalize(trial_DE_n)
        data = pre_process(trial_DE_normalized, base_DE_normalized)

        sio.savemat(result_dir +  file,
                    { "data": data, "valence_labels": valence_labels,
                     "arousal_labels": arousal_labels})

