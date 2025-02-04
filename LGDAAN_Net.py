import tensorflow as tf
from tensorflow.keras import layers
from feature_extractor import FeatureExtractor
from attention_module import AttentionModule
from domain_discriminator import TemDom, SpeDom, DomClass
from emotion_classifier import EmoClass

class LGDAAN_Net(tf.keras.Model):
    def __init__(self):
        super(LGDAAN_Net, self).__init__()

        # 特征提取器
        self.feature_extractor = FeatureExtractor()

        # 初始化注意力模块
        self.attention_module = AttentionModule(filters=128, attention_size=64, k=8)

        # 局部域鉴别器
        self.local_time_discriminator = TemDom()
        self.local_spectral_discriminator = SpeDom()

        # 全局域鉴别器
        self.global_discriminator = DomClass()

        # 情感分类器
        self.emotion_classifier = EmoClass()

    def call(self, source_time, source_spectral, target_time, target_spectral, source_domain_labels, target_domain_labels):
        # 特征提取
        source_time_features, target_time_features, source_spectral_features, target_spectral_features = self.feature_extractor(
            source_time, source_spectral, target_time, target_spectral)

        # 将源域时间特征和光谱特征进行拼接
        source_features = tf.concat([source_time_features, source_spectral_features], axis=-1)  # 拼接源域的时间和光谱特征
        target_features = tf.concat([target_time_features, target_spectral_features], axis=-1)  # 拼接目标域的时间和光谱特征

        # 先进行注意力融合
        # 将源域和目标域拼接后的特征输入到注意力模块中
        source_fused_features = self.attention_module(source_features)  # 对源域拼接后的特征进行融合
        target_fused_features = self.attention_module(target_features)  # 对目标域拼接后的特征进行融合

        # 局部域鉴别器（时间特征和光谱特征）
        local_time_preds = self.local_time_discriminator([source_time_features])
        local_spectral_preds = self.local_spectral_discriminator([source_spectral_features])

        # 全局域鉴别器
        global_preds = self.global_discriminator([source_fused_features])

        # 情感分类
        emotion_preds = self.emotion_classifier(source_fused_features)

        return emotion_preds, local_time_preds, local_spectral_preds, global_preds
