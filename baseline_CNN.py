import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. 词汇表定义 (Task 5 核心特征)
# ==========================================
VOCAB = ['circle', 'square', 'triangle', 'red', 'blue', 'green', 'yellow', 'black', 'above', 'below', 'small', 'medium',
         'big']


# ==========================================
# 2. Dataset 类定义
# ==========================================
class ShapeDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 确保列名与你的 labels.csv 一致 (0:image_id, 1:file_name, 2:description)
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['file_name'])
        image = Image.open(img_name).convert('RGB')

        desc = self.df.iloc[idx]['description'].lower()
        label = torch.zeros(len(VOCAB))
        for i, word in enumerate(VOCAB):
            if word in desc:
                label[i] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# 3. 成员 2 的基础 CNN 架构
# ==========================================
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 4. 主程序：数据划分、训练与评估
# ==========================================
def main():
    # A. 数据加载与划分
    if not os.path.exists('labels.csv'):
        print("错误：找不到 labels.csv，请检查路径。")
        return

    full_df = pd.read_csv('labels.csv')
    # 按照 80% 训练, 20% 测试划分
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    print(f"数据划分完成。训练集: {len(train_df)}, 测试集: {len(test_df)}")

    # B. 配置
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(ShapeDataset(train_df, 'images', transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(ShapeDataset(test_df, 'images', transform), batch_size=32, shuffle=False)

    model = BaselineCNN(len(VOCAB)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # C. 训练循环
    epochs = 10
    history = {'loss': []}

    print(f"开始在 {device} 上训练...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # D. 评估测试集效果
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()  # 阈值设为 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # 计算每个属性的准确率
    print("\n--- 测试集效果分析 ---")
    for i, word in enumerate(VOCAB):
        correct = (all_preds[:, i] == all_labels[:, i]).sum().item()
        accuracy = correct / len(all_labels) * 100
        print(f"属性 [{word:8}]: 准确率 {accuracy:.2f}%")

    # E. 绘制 Loss 曲线 (用于小组报告)
    plt.plot(range(1, epochs + 1), history['loss'])
    plt.title('Training Loss - Baseline Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('baseline_loss_plot.png')
    print("\nLoss 曲线已保存为 baseline_loss_plot.png")


if __name__ == '__main__':
    main()