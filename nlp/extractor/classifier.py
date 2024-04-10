"""
业务领域的分类器
"""
import pandas
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import f1_score, accuracy_score

# 将标签列转换为列表形式
df = pd.read_pickle('E:\Proj\Job_rec_sys_base_knowledge_graph\\nlp\dataset\process\process1.pkl')
df['业务领域'] = df['业务领域'].apply(lambda x: x.split(','))

# 使用MultiLabelBinarizer进行多热编码
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(df['业务领域'])

# 文本预处理（这里简化处理，实际使用时可以加入更复杂的文本预处理步骤）
tokenized_texts = [text.split() for text in df['职位']]

vocab = build_vocab_from_iterator(tokenized_texts, specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])


# 将文本转换为数字序列
def text_pipeline(x):
    return [vocab[token] for token in x]


# 将数据集分割为训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(tokenized_texts, encoded_labels, test_size=0.2)


# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text_pipeline(text), dtype=torch.long), torch.tensor(label, dtype=torch.float)


train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)


# 定义collate_fn来动态地填充批量中的文本数据
def collate_fn(batch):
    texts, labels = zip(*batch)  # 解压批次数据
    # 直接将texts列表中的每个元素转换为tensor，如果它们还不是tensor
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])  # 直接使用texts中的tensor，无需再次转换
    labels = torch.stack(labels)  # 堆叠标签张量
    return texts, labels


# 使用自定义的collate_fn来创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# 后续步骤（创建数据加载器、定义模型、训练等）与前面的示例相同。


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text).mean(dim=1)  # 注意这里改为dim=1以处理批次
        return self.activation(self.fc(embedded))


# 初始化模型
model = TextClassifier(len(vocab), embed_dim=100, num_classes=len(mlb.classes_))
# 定义损失函数和优化器
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# 函数：计算模型的性能指标
def calculate_metrics(preds, labels):
    # 预测标签（使用0.5作为阈值）
    pred_labels = preds > 0.5
    # 将预测和标签转换为与sklearn兼容的格式
    pred_labels = pred_labels.cpu().numpy()
    labels = labels.cpu().numpy()
    # 计算F1分数和准确率
    f1 = f1_score(labels, pred_labels, average='micro')
    accuracy = accuracy_score(labels, pred_labels.astype(int))
    return f1, accuracy


# 函数：评估模型
def evaluate_model(model, data_loader):
    model.eval()  # 将模型设置为评估模式
    total_f1 = 0
    total_accuracy = 0
    with torch.no_grad():  # 在评估阶段不计算梯度
        for texts, labels in data_loader:
            outputs = model(texts)
            # 使用sigmoid阈值来决定标签的预测
            f1, accuracy = calculate_metrics(torch.sigmoid(outputs), labels)
            total_f1 += f1
            total_accuracy += accuracy
    # 计算平均值
    avg_f1 = total_f1 / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    return avg_f1, avg_accuracy


# 训练循环
num_epochs = 10
# 训练循环中添加验证步骤
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # 训练结束后，在验证集上评估模型
    val_f1, val_accuracy = evaluate_model(model, test_loader)
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Val F1: {val_f1}, Val Accuracy: {val_accuracy}")

# 全部训练完成后，在测试集上评估模型
test_f1, test_accuracy = evaluate_model(model, test_loader)
print(f"Test F1: {test_f1}, Test Accuracy: {test_accuracy}")
