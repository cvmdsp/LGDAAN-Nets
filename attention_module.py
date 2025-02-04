import tensorflow as tf
from tensorflow.keras import layers


class AttentionModule(tf.keras.Model):
    def __init__(self, filters, attention_size, k=8):
        super(AttentionModule, self).__init__()

        # 定义卷积层
        self.conv_I = layers.Conv2D(filters=attention_size, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv_J = layers.Conv2D(filters=attention_size, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv_K = layers.Conv2D(filters=attention_size, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv_O = layers.Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', padding='same')

        # 可学习的 gamma 缩放因子，初始化为0
        self.gamma = tf.Variable(0.0, trainable=True, dtype=tf.float32)

        self.k = k  # 参数k用来调整J和K空间的大小，减小计算量

    def call(self, inputs):
        # 将输入映射到三个特征空间I(x), J(x), K(x)
        I_x = self.conv_I(inputs)  # I(x) = W_I * X
        J_x = self.conv_J(inputs)  # J(x) = W_J * X
        K_x = self.conv_K(inputs)  # K(x) = W_K * X

        # 进行矩阵运算计算s_ij
        K_x_T = tf.transpose(K_x, perm=[0, 2, 3, 1])  # 转置操作，方便后续计算
        s_ij = tf.matmul(K_x_T, J_x)  # s_ij = K(x)^T * J(x)

        # 激活：使用softmax对s_ij进行处理，得到attention map
        attention_map = tf.nn.softmax(s_ij, axis=-1)  # A_ij = softmax(s_ij)

        # 将注意力权重与I(x)相乘，得到加权后的特征
        weighted_I_x = tf.matmul(attention_map, I_x)  # 对每个位置的I(x)加权

        # 最后将加权后的结果映射到输出空间O(x)
        output = self.conv_O(weighted_I_x)  # Y_A = W_O * weighted_I_x

        # 添加 gamma 缩放因子，并返回最终融合的特征
        fused_output = self.gamma * output + inputs  # X = γ * Y_A + X

        return fused_output
