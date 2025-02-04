import tensorflow as tf
from tensorflow.keras import layers

class ConvLSTMBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, pool_size, pool_strides):
        super(ConvLSTMBlock, self).__init__()
        self.conv_lstm = layers.ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                           padding='same', return_sequences=True)
        self.batch_norm = layers.BatchNormalization()
        self.pooling = layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')

    def call(self, inputs):
        # 确保输入是5D张量（(batch_size, time_steps, height, width, channels)）
        if len(inputs.shape) == 4:
            # 如果输入是4D张量，则在第二维（时间步）上添加一个额外的维度
            inputs = tf.expand_dims(inputs, axis=1)  # 将其转换为5D张量

        x = self.conv_lstm(inputs)
        x = self.batch_norm(x)

        # 如果输入是5D张量，池化操作需要4D张量
        if len(x.shape) == 5:
            x = tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]])  # Reshaping to 4D for pooling

        x = self.pooling(x)
        return x

class TemporalExtractor(tf.keras.Model):
    def __init__(self):
        super(TemporalExtractor, self).__init__()
        self.cl_block1 = ConvLSTMBlock(filters=64, kernel_size=(4, 4), strides=(2, 2), pool_size=(2, 2), pool_strides=(2, 2))
        self.cl_block2 = ConvLSTMBlock(filters=128, kernel_size=(4, 4), strides=(2, 2), pool_size=(2, 2), pool_strides=(2, 2))
        self.cl_block3 = ConvLSTMBlock(filters=128, kernel_size=(4, 4), strides=(2, 2), pool_size=(2, 2), pool_strides=(2, 2))

    def call(self, inputs):
        x = self.cl_block1(inputs)
        x = self.cl_block2(x)
        x = self.cl_block3(x)
        return x

class SpectralExtractor(tf.keras.Model):
    def __init__(self):
        super(SpectralExtractor, self).__init__()
        self.cl_block1 = ConvLSTMBlock(filters=64, kernel_size=(4, 4), strides=(2, 2), pool_size=(2, 2), pool_strides=(2, 2))
        self.cl_block2 = ConvLSTMBlock(filters=128, kernel_size=(4, 4), strides=(2, 2), pool_size=(2, 2), pool_strides=(2, 2))
        self.cl_block3 = ConvLSTMBlock(filters=128, kernel_size=(4, 4), strides=(2, 2), pool_size=(2, 2), pool_strides=(2, 2))

    def call(self, inputs):
        x = self.cl_block1(inputs)
        x = self.cl_block2(x)
        x = self.cl_block3(x)
        return x

class FeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.temporal_extractor = TemporalExtractor()
        self.spectral_extractor = SpectralExtractor()

    def call(self, source_time, source_spectral, target_time, target_spectral):
        source_time_features = self.temporal_extractor(source_time)
        target_time_features = self.temporal_extractor(target_time)
        source_spectral_features = self.spectral_extractor(source_spectral)
        target_spectral_features = self.spectral_extractor(target_spectral)
        return source_time_features, target_time_features, source_spectral_features, target_spectral_features
