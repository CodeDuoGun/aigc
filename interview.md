# transformer

# 经典模型深度理解

    RNN/LSTM：门控机制数学推导、梯度消失解决方案（如GRU的简化设计）

    Transformer：复杂度分析（Flash Attention优化）、位置编码的多种实现方式（RoPE、ALiBi）

# 预训练模型：

    BERT家族（DeBERTav3的 disentangled attention）

    GPT系列（稀疏注意力、MoE架构）

    T5框架的文本到文本统一范式

    多模态模型（CLIP、Flamingo）
# 推理优化
ONNX模型转换、TensorRT加速、大模型服务框架（vLLM/Text Generation Inference），triton cuda核

# 模型压缩：知识蒸馏（MiniLMv2）、量化（GPTQ）、参数高效微调（LoRA）
    知识蒸馏
    量化
    lora

# 微调

1、大模型Tuning 方法
    全参微调
    低参微调：
        1、Prompt Tuning， 在Prompt部分由若干可训练的Token组成，作为输入文本的前缀，在训练过程中只有这些Token的Embeddings被训练，
        2、P-Tuning ，将Prompt转换为可学习的Embedding层，并用MLP+LSTM处理
        3、P-Tuning v2，在多层加入Prompt tokens，增加可学习参数数量，
        4、Prefix Tuning， 在输入前添加可学习的virtual tokens作为Prefix，训练时仅更新Prefix参数，Transformer其他部分固定
        5、Adapter Tuning，预训练模型的基础上添加额外的适配器模块，这些适配器模块是可训练的，而原始模型的参数保持不变。适配器模块可以是简单的线性层、前馈神经网络等结构，用于对模型的输出进行调整，使其更好地适应特定任务
        6、LoRA，在模型的决定性层次中引入小型、低秩的矩阵来实现模型行为的微调，
        7、BitFit,
    
2、介绍损失函数

3、如何把当前任务prompt化？
    1、定义任务： 分类、生成、情感分析等
    2、设计prompt：******
        模板化（）、填空式（该新闻属于___类别）、问答式（大数据的主要特点是什么？回答：）、指令式（请给出这段文字的主要观点）
    3、生成输出： 
        将组合后的输入文本输入大模型，生成模型输出
    4、优化prompt：
        多次试验和迭代： 
        使用任务特定关键词：积极/消极
        提供示例：
        组合式prompt：组合多个prompt
        少样本示例：
        动态prompt调整： 基于生成的初始结果实时调整Prompt内容，动态优化Prompt
4、写自注意力的公式，解释
    
5、大kernel的优势，是不是越大越好？
    ERF理论：大Kernel可以在较少的层数下获得较大的感受野，有助于模型更好地理解图像的整体结构和上下文信息。
    局限性：
        计算量和参数量大：
        忽略局部细节：大kernel 捕获全局信息
        训练难度大：
6、transformer head 为什么要降维？
    （CNN 多核思想是什么：并行处理 多通道处理）
    1、增强模型表达能力：高维
    
7、transformer 为什么要用三个不一样的QKV?
    QKV 维度相同，
    Q：
    K：
    V：
    
8、NMS及其变体：
    NMS步骤： 
        1、排序：将所有检测框（bounding boxes）按照置信度（confidence score）从高到低进行排序。
        2、选择最高置信度的框：选择置信度最高的检测框作为当前的最优框。
        3、计算IoU：计算该最优框与其余检测框的交并比（IoU，Intersection over Union）。
        4、抑制重叠框：将IoU大于某个阈值（通常为0.5）的检测框抑制掉，即从候选框中移除。
        5、重复上述步骤：重复上述步骤，直到所有检测框都被处理完毕
    NMS的变体
        1、Soft NMS：根据IoU值对重叠框的置信度进行调整
        2、softer NMS：会根据IoU值和置信度的组合来调整重叠框的置信度，
        3、Weighted NMS：会根据IoU值和置信度的加权和来调整重叠框的置信度，从而更合理地保留检测结果
        4、DIoU NMS：Distance 会根据IoU值和中心点距离的组合来调整重叠框的置信度
        5、CIoU NMS：Complete 根据IoU值、中心点距离和长宽比的组合来调整重叠框的置信度。。。
9、SVM？

10、权重初始化方法？
    零初始化
    随机初始化
    预训练初始化
    正交初始化
    
10-1、 梯度消失？

10-2、 梯度扩散？

10-3、 优化器？

10-4、 精度？

11、为什么transformer 要用layerNorm？

12、为什么 self-attention 可以堆叠多层，有什么作用？

13、常用的聚类算法？kmeans，dbscan，mean shift

14、clip vae t5 区别


