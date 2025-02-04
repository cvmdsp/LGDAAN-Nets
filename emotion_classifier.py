import tensorflow as tf
from tensorflow.keras import layers

class EmoClass(tf.keras.Model):
    def __init__(self):
        super(EmoClass, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.output_layer(x)
