import scipy.io as sio
import os
import numpy as np

np.random.seed(0)


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
    # return shape:9*9
    return data_2D


def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[1]):
        norm_dataset_1D[:, i] = feature_normalize(dataset_1D[:, i])
    # return shape: m*32
    return norm_dataset_1D


def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    # return shape: 9*9
    return data_normalized


def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D


def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
    # return shape: m*9*9
    return norm_dataset_2D


def windows(data, size):
    start = 0
    while ((start + size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size


def segment_signal_without_transition(data, window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if ((len(data[start:end]) == window_size)):
            if (start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

            else:
                segments = np.vstack([segments, data[start:end]])

    return segments


def apply_mixup(dataset_file, window_size):  # initial empty label arrays
    print("Processing", dataset_file, "..........")
    data_file_in = sio.loadmat(dataset_file)
    data_in = data_file_in["data"].transpose(0, 2, 1)
    valence_labels = data_file_in["labels"][:, 0] > 5
    arousal_labels = data_file_in["labels"][:, 1] > 5
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    for i in range(len(valence_labels)):
        for j in range(0, 60):
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
    print("labels:", final_arousal_labels.shape)
    data_inter = np.empty([0, window_size, 9, 9])
    trials = data_in.shape[0]

    # Data pre-processing
    for trial in range(0, trials):
        base_signal = (data_in[trial, 0:128, 0:32] + data_in[trial, 128:256, 0:32] + data_in[trial, 256:384, 0:32]) / 3
        data = data_in[trial, 384:8064, 0:32]
        # compute the deviation between baseline signals and experimental signals
        for i in range(0, 60):
            data[i * 128:(i + 1) * 128, 0:32] = data[i * 128:(i + 1) * 128, 0:32] - base_signal
        label_index = trial
        # read data and label
        data = norm_dataset(data)
        data = segment_signal_without_transition(data,  window_size)
        data = dataset_1Dto2D(data)
        data = data.reshape(int(data.shape[0] / window_size), window_size, 9, 9)
        # append new data and label
        data_inter = np.vstack([data_inter, data])


    print("total data size:", data_inter.shape)

    return data_inter, final_arousal_labels, final_valence_labels

if __name__ == '__main__':
    dataset_dir = "./DEAP_data/"
    window_size = 128
    output_dir = "./DEAP_temp/"
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
        # get directory name for one subject
    for task in os.listdir(dataset_dir):  # 输出该路径下的所有文件名称
        print("processing: ", task, "......")
        file_path = os.path.join(dataset_dir, task)  # 连接两个或更多的路径名组件

        file = os.path.join(dataset_dir, task)
        shuffled_data, final_arousal_labels, final_valence_labels = apply_mixup(file, window_size)

        sio.savemat(output_dir + task,
                        {"data": shuffled_data, "valence_labels": final_valence_labels,
                     "arousal_labels": final_arousal_labels})

