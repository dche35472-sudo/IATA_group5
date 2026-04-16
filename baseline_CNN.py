import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Vocabulary Definition
# ==========================================
VOCAB = ['circle', 'square', 'triangle', 'red', 'blue', 'green', 'yellow', 'black',
         'above', 'below', 'left', 'right', 'small', 'medium', 'big']


# ==========================================
# 2. Advanced Logic-based Sentence Generator
# ==========================================
def generate_reciprocal_sentences(tags):
    shapes = [t for t in tags if t in ['circle', 'square', 'triangle']]
    colors = [t for t in tags if t in ['red', 'blue', 'green', 'yellow', 'black']]
    sizes = [t for t in tags if t in ['small', 'medium', 'big']]
    relations = [t for t in tags if t in ['above', 'below', 'left', 'right']]

    sentences = []
    # Logic: Only pair if at least two shapes are detected
    if len(shapes) >= 2:
        # Match attributes to objects (simplistic index-based pairing)
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
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes), nn.Sigmoid()
        )

    def forward(self, x): return self.classifier(self.features(x))


# ==========================================
# 4. Training and Evaluation Pipeline
# ==========================================
def main():
    if not os.path.exists('labels.csv'):
        print("Error: labels.csv not found.")
        return

    # FULL DATASET LOAD
    df = pd.read_csv('labels.csv')
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(ShapeDataset(train_df, 'images', transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(ShapeDataset(test_df, 'images', transform), batch_size=1, shuffle=False)

    model = BaselineCNN(len(VOCAB)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 1. Training Loop
    epochs = 15
    loss_history = []
    print(f"Training on {len(train_df)} samples using {device}...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, lbls, _, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

    # 2. Visualizing Training Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title('Final Model Loss Curve')
    plt.savefig('final_loss_curve.png')

    # 3. Final Inference
    model.eval()
    all_preds, all_labels, results = [], [], []

    print("Generating final predictions...")
    with torch.no_grad():
        for imgs, lbls, fnames, descs in test_loader:
            imgs = imgs.to(device)
            raw_out = model(imgs)
            preds_bin = (raw_out > 0.5).float().cpu()

            all_preds.append(preds_bin.numpy())
            all_labels.append(lbls.numpy())

            tags = [VOCAB[j] for j, v in enumerate(preds_bin[0]) if v == 1.0]
            results.append({
                'file_name': fnames[0],
                'ground_truth': descs[0],
                'baseline_output': generate_reciprocal_sentences(tags)
            })

    # 4. Save CSV & Metrics
    pd.DataFrame(results).to_csv('baseline_final_results.csv', index=False)

    # 5. Confusion Matrices
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    for i, class_name in enumerate(VOCAB):
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes.flatten()[i], cmap='Greens', cbar=False)
        axes.flatten()[i].set_title(class_name)
    plt.tight_layout()
    plt.savefig('final_confusion_matrices.png')

    print(
        "Full process complete. Output: baseline_final_results.csv, final_loss_curve.png, final_confusion_matrices.png")


if __name__ == '__main__':
    main()