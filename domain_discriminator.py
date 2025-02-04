import tensorflow as tf
from tensorflow.keras import layers

@tf.custom_gradient
def gradient_reversal(x, alpha=1.0):
    def grad(dy):
        return -dy * alpha, None
    return x, grad

class GradientReversalLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)

    def call(self, x, alpha=1.0):
        return gradient_reversal(x, alpha)

class DomainDiscriminator(tf.keras.Model):
    def __init__(self, output_size):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(output_size, activation='softmax')
        self.grl = GradientReversalLayer()

    def call(self, inputs, alpha=1.0):
        x = self.grl(inputs, alpha)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output_layer(x)

class DomClass(DomainDiscriminator):
    def __init__(self):
        super(DomClass, self).__init__(output_size=2)

class TemDom(DomainDiscriminator):
    def __init__(self):
        super(TemDom, self).__init__(output_size=2)

class SpeDom(DomainDiscriminator):
    def __init__(self):
        super(SpeDom, self).__init__(output_size=2)
