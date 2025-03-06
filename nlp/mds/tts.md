# 语音合成过程 tts
1、数据预处理：文本归一化（Text Normalization）、音素对齐（Phoneme Alignment）、语音分段与降噪。

2、特征提取：梅尔频谱、基频（F0）、能量（Energy）、音素时长（Phoneme Duration）

# Librosa 用途
# 
Zero-Shot/少样本合成：如何通过少量语音克隆目标音色（如YourTTS、Vall-E）。
情感与风格控制：在频谱或隐空间中引入风格嵌入（Style Embedding）。
多语言与跨语言合成：统一音素表（如IPA）、语言对抗训练。
实时性与轻量化：知识蒸馏（如将Tacotron2蒸馏到FastSpeech）、量化与剪枝

