import tensorflow as tf
from tensorflow.keras import layers
from feature_extractor import FeatureExtractor
from attention_module import AttentionModule
from domain_discriminator import TemDom, SpeDom, DomClass
from emotion_classifier import EmoClass

class LGDAAN_Net(tf.keras.Model):
    def __init__(self):
        super(LGDAAN_Net, self).__init__()

        
        self.feature_extractor = FeatureExtractor()

        
        self.attention_module = AttentionModule(filters=128, attention_size=64, k=8)

       
        self.local_time_discriminator = TemDom()
        self.local_spectral_discriminator = SpeDom()

        
        self.global_discriminator = DomClass()

        
        self.emotion_classifier = EmoClass()

    def call(self, source_time, source_spectral, target_time, target_spectral, source_domain_labels, target_domain_labels):
        
        source_time_features, target_time_features, source_spectral_features, target_spectral_features = self.feature_extractor(
            source_time, source_spectral, target_time, target_spectral)

        
        source_features = tf.concat([source_time_features, source_spectral_features], axis=-1)  
        target_features = tf.concat([target_time_features, target_spectral_features], axis=-1)  

        
        source_fused_features = self.attention_module(source_features)  
        target_fused_features = self.attention_module(target_features)  

        
        local_time_preds = self.local_time_discriminator([source_time_features])
        local_spectral_preds = self.local_spectral_discriminator([source_spectral_features])

        
        global_preds = self.global_discriminator([source_fused_features])

        
        emotion_preds = self.emotion_classifier(source_fused_features)

        return emotion_preds, local_time_preds, local_spectral_preds, global_preds
