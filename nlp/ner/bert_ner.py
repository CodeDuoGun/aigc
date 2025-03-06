import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# 配置
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 标签到 ID 的映射
LABEL_TO_ID = {
    'O': 0,
    'B-LOC': 1,  # 位置
    'I-LOC': 2,
    'B-PER': 3,  # 人名
    'I-PER': 4,
    'B-ORG': 5,  # 组织
    'I-ORG': 6,
    # 其他标签可以根据需求添加
}

# 数据加载
class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 将标签对齐到 token
        # 这里假设输入的文本已经被分词并与标签对齐
        # 如果不是，需要复杂的对齐处理，这里暂时简化
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label[:self.max_len], dtype=torch.long)
        }

# 模型
class BERTForNER(nn.Module):
    def __init__(self, num_labels):
        super(BERTForNER, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# 训练
def train(model, train_loader, val_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            model.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {total_loss / len(train_loader)}")

        # 验证
        model.eval()
        val_loss = 0
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                loss, logits = model(input_ids, attention_mask, labels)
                val_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = labels.cpu().numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend([list(l) for l in label_ids])

        # 去除 Pad 和特殊 token 的结果
        pred_list, true_list = [], []
        for p, l in zip(predictions, true_labels):
            pred = [LABEL_TO_ID[i] for i in p if i != LABEL_TO_ID['O']]
            true = [LABEL_TO_ID[i] for i in l if i != LABEL_TO_ID['O']]
            pred_list.append(pred)
            true_list.append(true)

        print(f"Validation Loss: {val_loss / len(val_loader)}")
        print(f"Validation F1 Score: {f1_score(true_list, pred_list)}")

# 测试
def test(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            _, logits = model(input_ids, attention_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend([list(l) for l in label_ids])

    # 去除 Pad 和特殊 token 的结果
    pred_list, true_list = [], []
    for p, l in zip(predictions, true_labels):
        pred = [LABEL_TO_ID[i] for i in p if i != LABEL_TO_ID['O']]
        true = [LABEL_TO_ID[i] for i in l if i != LABEL_TO_ID['O']]
        pred_list.append(pred)
        true_list.append(true)

    print(f"F1 Score: {f1_score(true_list, pred_list)}")
    print(f"Precision: {precision_score(true_list, pred_list)}")
    print(f"Recall: {recall_score(true_list, pred_list)}")
    print(classification_report(true_list, pred_list))

# 主函数
def main():
    # 假设我们有一组训练数据
    # 这里需要替换为实际的数据加载代码
    texts = ["这是一句测试文本", "欢迎来到上海"]
    labels = [
        [LABEL_TO_ID['O'], LABEL_TO_ID['O'], LABEL_TO_ID['O'], LABEL_TO_ID['O'], LABEL_TO_ID['O']],
        [LABEL_TO_ID['O'], LABEL_TO_ID['O'], LABEL_TO_ID['B-LOC'], LABEL_TO_ID['I-LOC'], LABEL_TO_ID['O']]
    ]

    # 划分数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 分词器
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    # 数据加载器
    train_dataset = NERDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = NERDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = NERDataset(val_texts, val_labels, tokenizer, MAX_LEN)  # 测试集用验证集代替

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    model = BERTForNER(num_labels=len(LABEL_TO_ID)).to(DEVICE)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练
    train(model, train_loader, val_loader, optimizer, EPOCHS)

    # 测试
    test(model, test_loader)

if __name__ == "__main__":
    main()