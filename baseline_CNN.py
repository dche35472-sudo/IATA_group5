import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# 1. 建立词汇表 (根据 Task 5 的描述特征)
VOCAB = ['circle', 'square', 'triangle', 'red', 'blue', 'green', 'yellow', 'black', 'above', 'below', 'small', 'medium',
         'big']


class ShapeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 加载图像
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')

        # 标签处理：多标签 One-hot
        # 比如描述里有 "red circle above blue square"，对应词汇表位置设为 1
        desc = self.df.iloc[idx, 2].lower()
        label = torch.zeros(len(VOCAB))
        for i, word in enumerate(VOCAB):
            if word in desc:
                label[i] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, label


# 2. 成员 2 的设计：基础 CNN 架构
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        # 卷积层：提取视觉特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32 -> 16
        )
        # 全连接层：预测属性
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # 多标签分类关键
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# 3. 实验配置
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 实例化并准备训练
dataset = ShapeDataset(csv_file='labels.csv', img_dir='images', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = BaselineCNN(len(VOCAB))
criterion = nn.BCELoss()  # 多标签分类常用 Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 建议在训练循环中记录 Loss，用于后期画图
print("Baseline 准备就绪，开始训练...")