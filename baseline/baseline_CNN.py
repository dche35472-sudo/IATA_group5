import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Vocabulary Definition
# ==========================================
VOCAB = ['circle', 'square', 'triangle', 'red', 'blue', 'green', 'yellow', 'black',
         'above', 'below', 'left', 'right', 'overlapping', 'small', 'medium', 'big']


# ==========================================
# 2. Logic-based Sentence Generator
# ==========================================
def generate_reciprocal_sentences(tags):
    shapes = [t for t in tags if t in ['circle', 'square', 'triangle']]
    colors = [t for t in tags if t in ['red', 'blue', 'green', 'yellow', 'black']]
    sizes = [t for t in tags if t in ['small', 'medium', 'big']]
    relations = [t for t in tags if t in ['above', 'below', 'left', 'right', 'overlapping']]

    sentences = []
    if len(shapes) >= 2:
        obj1 = f"{sizes[0] if sizes else ''} {colors[0] if colors else ''} {shapes[0]}".strip()
        obj2 = f"{sizes[1] if len(sizes) > 1 else ''} {colors[1] if len(colors) > 1 else ''} {shapes[1]}".strip()

        for rel in relations:
            if rel == 'above':
                sentences.append(f"a {obj1} is above a {obj2}")
                sentences.append(f"a {obj2} is below a {obj1}")
            elif rel == 'below':
                sentences.append(f"a {obj1} is below a {obj2}")
                sentences.append(f"a {obj2} is above a {obj1}")
            elif rel == 'left':
                sentences.append(f"a {obj1} is left of a {obj2}")
                sentences.append(f"a {obj2} is right of a {obj1}")
            elif rel == 'right':
                sentences.append(f"a {obj1} is right of a {obj2}")
                sentences.append(f"a {obj2} is left of a {obj1}")
            elif rel == 'overlapping':
                sentences.append(f"a {obj1} is overlapping a {obj2}")
                sentences.append(f"a {obj2} is overlapping a {obj1}")

    elif len(shapes) == 1:
        obj = f"{sizes[0] if sizes else ''} {colors[0] if colors else ''} {shapes[0]}".strip()
        sentences.append(f"there is a {obj}")

    unique_sents = list(dict.fromkeys(sentences))
    return " | ".join(unique_sents) if unique_sents else "no objects detected"


# ==========================================
# 3. Dataset & Model Architecture
# ==========================================
class ShapeDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.df.iloc[idx]['file_name']))
        image = Image.open(img_name).convert('RGB')
        desc = str(self.df.iloc[idx]['description']).lower()
        label = torch.zeros(len(VOCAB))
        for i, word in enumerate(VOCAB):
            if word in desc: label[i] = 1.0
        if self.transform: image = self.transform(image)
        return image, label, self.df.iloc[idx]['file_name'], self.df.iloc[idx]['description']


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Feature Extraction: 4 Convolutional Layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        # Classification: 2 Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes), nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==========================================
# 4. Pipeline with Accuracy Calculation
# ==========================================
def main():
    if not os.path.exists('../labels.csv'):
        print("Error: labels.csv not found.")
        return

    df = pd.read_csv('../labels.csv')
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(ShapeDataset(train_df, '../images', transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(ShapeDataset(test_df, '../images', transform), batch_size=64, shuffle=False)

    model = BaselineCNN(len(VOCAB)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print(f"Starting training on {device}...")
    for epoch in range(15):
        model.train()
        running_loss = 0.0
        for imgs, lbls, _, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/15], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation phase
    model.eval()
    all_results = []
    correct_samples = 0
    total_samples = 0

    print("Evaluating model...")
    with torch.no_grad():
        for imgs, lbls, fnames, descs in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            preds_bin = (outputs > 0.5).float()

            # Calculate Exact Match Accuracy (Subset Accuracy)
            for i in range(imgs.size(0)):
                total_samples += 1
                if torch.equal(preds_bin[i], lbls[i]):
                    correct_samples += 1

                # Collect data for CSV
                tags = [VOCAB[j] for j, v in enumerate(preds_bin[i]) if v == 1.0]
                all_results.append({
                    'file_name': fnames[i],
                    'ground_truth': descs[i],
                    'baseline_output': generate_reciprocal_sentences(tags)
                })

    accuracy = (correct_samples / total_samples) * 100
    print(f"\nFinal Test Accuracy (Exact Match): {accuracy:.2f}%")

    pd.DataFrame(all_results).to_csv('baseline_overlapping_results.csv', index=False)
    print("Results saved to baseline_overlapping_results.csv")


if __name__ == '__main__':
    main()