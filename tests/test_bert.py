import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese").to(device)


# 函数：使用BERT模型提取文本特征
def extract_features(texts, model, tokenizer, device):
    model.eval()  # 将模型设置为评估模式
    features = []

    for text in tqdm(texts, desc="Extracting features"):
        # 确保输入数据也在GPU上
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]  # 通常是元组的第一个元素
        # 计算平均隐藏状态并转移到CPU
        mean_hidden_states = hidden_states.mean(dim=1).cpu()
        features.append(mean_hidden_states)

    return torch.cat(features, dim=0)


if __name__ == '__main__':
    df = pd.read_csv("data/text.csv")
    texts = df['text'].to_list()
    features = extract_features(texts, model, tokenizer, device)
    np.save('data/features.npy', features.numpy())