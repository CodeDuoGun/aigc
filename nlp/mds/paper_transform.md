# paper
1、 AutoGLM 智谱AI 智能体 https://xiao9905.github.io/AutoGLM/
2、 CLIP 文本编码器 https://github.com/openai/CLIP
3、 AutoGPT ai 和智能体的结合 https://github.com/Significant-Gravitas/AutoGPT
4、
# GPT

# BERT


# Transformer
    1、编码器

    2、解码器

    3、注意力
    
        缩放点积注意力：
            Attention(Q, K, V) = softmax(QKT / √dk) V 
            虽然对于小的dk值，两种机制的表现相似，但对于较大的dk值，加性注意力优于未缩放的点积注意力[3]。我们怀疑，对于大的dk值，点积的大小会变得很大，将softmax函数推入具有极小梯度的区域。为了解决这个问题，我们将点积缩放为√1/dk。
        多头注意力：三个地方（）
    4、逐点前馈神经网络
        除了注意力子层外，编码器和解码器中的每一层都包含一个全连接前馈网络，
        该网络对每个位置单独且相同地应用。这包括两个线性变换，中间有ReLU激活。
        FFN(x) = max(0, xW1 + b1)W2 + b2 (2)
        虽然不同位置的线性变换相同，但在不同层使用不同的参数。
        可以视为卷积，卷积核大小为1。输入和输出的维度为dmodel = 512，内层的维度为df f = 2048


# 