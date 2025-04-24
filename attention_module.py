import tensorflow as tf
from tensorflow.keras import layers


class AttentionModule(tf.keras.Model):
    def __init__(self, filters, attention_size, k=8):
        super(AttentionModule, self).__init__()

        
        self.conv_I = layers.Conv2D(filters=attention_size, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv_J = layers.Conv2D(filters=attention_size, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv_K = layers.Conv2D(filters=attention_size, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv_O = layers.Conv2D(filters=filters, kernel_size=(1, 1), activation='relu', padding='same')

        
        self.gamma = tf.Variable(0.0, trainable=True, dtype=tf.float32)

        self.k = k  

    def call(self, inputs):
        I_x = self.conv_I(inputs)  
        J_x = self.conv_J(inputs)  
        K_x = self.conv_K(inputs)  

        
        K_x_T = tf.transpose(K_x, perm=[0, 2, 3, 1])  
        s_ij = tf.matmul(K_x_T, J_x)  

        
        attention_map = tf.nn.softmax(s_ij, axis=-1)  

        
        weighted_I_x = tf.matmul(attention_map, I_x)  

        
        output = self.conv_O(weighted_I_x)  

        
        fused_output = self.gamma * output + inputs  

        return fused_output
