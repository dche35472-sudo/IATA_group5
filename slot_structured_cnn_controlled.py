
import json
import math
import random
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================================================
# Controlled slot-based model
# Goal: keep the encoder aligned with the updated baseline,
# while changing only the output representation.
#
# Alignment with updated baseline:
# - same 4-layer CNN encoder style
# - same 64x64 image size
# - same train/test split style (test_size=0.15, random_state=42)
# - slot-based output heads instead of bag-of-words multi-label output
# =========================================================


def set_seed(seed: int = 42):
    """Make training runs more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RELATION_CANONICAL = {
    "left of": "left of",
    "right of": "right of",
    "above": "above",
    "below": "below",
    "overlapping": "overlapping",
}

HEAD_SPECS = {
    "anchor_size": ["small", "medium", "big"],
    "anchor_color": ["red", "blue", "green", "yellow", "black"],
    "anchor_shape": ["circle", "square", "triangle"],
    "rel1": ["left of", "right of", "above", "below", "overlapping"],
    "target1_size": ["small", "medium", "big"],
    "target1_color": ["red", "blue", "green", "yellow", "black"],
    "target1_shape": ["circle", "square", "triangle"],
    "rel2": ["left of", "right of", "above", "below", "overlapping"],
    "target2_size": ["small", "medium", "big"],
    "target2_color": ["red", "blue", "green", "yellow", "black"],
    "target2_shape": ["circle", "square", "triangle"],
}

HEAD_ORDER = list(HEAD_SPECS.keys())


DESCRIPTION_RE = re.compile(
    r"^a (?P<anchor_size>small|medium|big) "
    r"(?P<anchor_color>red|blue|green|yellow|black) "
    r"(?P<anchor_shape>circle|square|triangle) "
    r"is (?P<rel1>left of|right of|above|below|overlapping) a "
    r"(?P<target1_size>small|medium|big) "
    r"(?P<target1_color>red|blue|green|yellow|black) "
    r"(?P<target1_shape>circle|square|triangle) \| "
    r"a (?P=anchor_size) (?P=anchor_color) (?P=anchor_shape) "
    r"is (?P<rel2>left of|right of|above|below|overlapping) a "
    r"(?P<target2_size>small|medium|big) "
    r"(?P<target2_color>red|blue|green|yellow|black) "
    r"(?P<target2_shape>circle|square|triangle)$"
)


def parse_description(desc: str) -> dict:
    desc = str(desc).strip().lower()
    m = DESCRIPTION_RE.match(desc)
    if not m:
        raise ValueError(f"Could not parse description: {desc}")
    parsed = dict(m.groupdict())
    parsed["rel1"] = RELATION_CANONICAL[parsed["rel1"]]
    parsed["rel2"] = RELATION_CANONICAL[parsed["rel2"]]
    return parsed


def build_sentence_from_slots(slot_values: dict) -> str:
    return (
        f"a {slot_values['anchor_size']} {slot_values['anchor_color']} {slot_values['anchor_shape']} "
        f"is {slot_values['rel1']} a {slot_values['target1_size']} {slot_values['target1_color']} {slot_values['target1_shape']} | "
        f"a {slot_values['anchor_size']} {slot_values['anchor_color']} {slot_values['anchor_shape']} "
        f"is {slot_values['rel2']} a {slot_values['target2_size']} {slot_values['target2_color']} {slot_values['target2_shape']}"
    )


class SimpleImageTransform:
    def __init__(self, image_size: int = 64):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr, dtype=torch.float32)


class SlotDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform or SimpleImageTransform(64)

        parsed_rows = []
        for desc in self.df["description"]:
            parsed_rows.append(parse_description(desc))
        self.parsed_df = pd.DataFrame(parsed_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parsed = self.parsed_df.iloc[idx]

        image_path = os.path.join(self.img_dir, str(row["file_name"]))
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_dict = {}
        for head_name, classes in HEAD_SPECS.items():
            value = parsed[head_name]
            label_dict[head_name] = classes.index(value)

        return {
            "image": image,
            "labels": label_dict,
            "file_name": row["file_name"],
            "description": row["description"],
        }


def collate_batch(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    file_names = [item["file_name"] for item in batch]
    descriptions = [item["description"] for item in batch]

    labels = {}
    for head_name in HEAD_ORDER:
        labels[head_name] = torch.tensor(
            [item["labels"][head_name] for item in batch], dtype=torch.long
        )

    return {
        "image": images,
        "labels": labels,
        "file_name": file_names,
        "description": descriptions,
    }


class BaselineAlignedEncoder(nn.Module):
    """
    Encoder aligned with the updated baseline_CNN.py:
    4 conv layers + maxpool, then flatten -> linear(256*4*4, 512) -> relu -> dropout
    Assumes 64x64 input.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return x


class ControlledSlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BaselineAlignedEncoder()
        self.heads = nn.ModuleDict({
            head_name: nn.Linear(512, len(classes))
            for head_name, classes in HEAD_SPECS.items()
        })

    def forward(self, x):
        z = self.encoder(x)
        return {head_name: head(z) for head_name, head in self.heads.items()}


def compute_loss(outputs: dict, labels: dict, criterion) -> torch.Tensor:
    losses = [criterion(outputs[head_name], labels[head_name]) for head_name in HEAD_ORDER]
    return sum(losses) / len(losses)


def predict_slot_values(outputs: dict) -> dict:
    pred_values = {}
    for head_name, logits in outputs.items():
        pred_idx = logits.argmax(dim=1).cpu().numpy().tolist()
        pred_values[head_name] = [HEAD_SPECS[head_name][i] for i in pred_idx]
    return pred_values


def labels_to_slot_values(labels: dict) -> dict:
    gold_values = {}
    for head_name, idx_tensor in labels.items():
        gold_values[head_name] = [HEAD_SPECS[head_name][int(i)] for i in idx_tensor.cpu().numpy().tolist()]
    return gold_values


def evaluate_model(model, loader, device):
    model.eval()

    total_samples = 0
    per_head_correct = {head: 0 for head in HEAD_ORDER}
    exact_sentence_correct = 0
    joint_slots_correct = 0

    all_rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            outputs = model(images)

            pred_values = predict_slot_values(outputs)
            gold_values = labels_to_slot_values(labels)

            batch_size = images.size(0)
            total_samples += batch_size

            for i in range(batch_size):
                pred_slots = {head: pred_values[head][i] for head in HEAD_ORDER}
                gold_slots = {head: gold_values[head][i] for head in HEAD_ORDER}

                head_matches = {}
                all_match = True
                for head in HEAD_ORDER:
                    match = int(pred_slots[head] == gold_slots[head])
                    head_matches[head] = match
                    per_head_correct[head] += match
                    if not match:
                        all_match = False

                pred_sentence = build_sentence_from_slots(pred_slots)
                gold_sentence = build_sentence_from_slots(gold_slots)
                exact_match = int(pred_sentence == gold_sentence)

                if all_match:
                    joint_slots_correct += 1
                if exact_match:
                    exact_sentence_correct += 1

                row = {
                    "file_name": batch["file_name"][i],
                    "gold_sentence": gold_sentence,
                    "pred_sentence": pred_sentence,
                    "exact_sentence_match": exact_match,
                    "all_slots_joint_match": int(all_match),
                }
                for head in HEAD_ORDER:
                    row[f"gold_{head}"] = gold_slots[head]
                    row[f"pred_{head}"] = pred_slots[head]
                    row[f"match_{head}"] = head_matches[head]
                all_rows.append(row)

    per_head_acc = {
        f"acc_{head}": per_head_correct[head] / total_samples for head in HEAD_ORDER
    }

    relation_accuracy = (
        per_head_correct["rel1"] + per_head_correct["rel2"]
    ) / (2 * total_samples)

    attribute_heads = [
        "anchor_size", "anchor_color", "anchor_shape",
        "target1_size", "target1_color", "target1_shape",
        "target2_size", "target2_color", "target2_shape",
    ]
    attribute_accuracy = sum(per_head_correct[h] for h in attribute_heads) / (
        len(attribute_heads) * total_samples
    )

    metrics = {
        "exact_sentence_accuracy": exact_sentence_correct / total_samples,
        "all_slots_joint_accuracy": joint_slots_correct / total_samples,
        "mean_slot_accuracy": sum(per_head_correct.values()) / (len(HEAD_ORDER) * total_samples),
        "relation_accuracy": relation_accuracy,
        "attribute_accuracy": attribute_accuracy,
    }
    metrics.update(per_head_acc)

    details_df = pd.DataFrame(all_rows)
    return metrics, details_df


def main():
    set_seed(42)

    if not os.path.exists("labels.csv"):
        raise FileNotFoundError("labels.csv not found in the current directory.")
    if not os.path.exists("images"):
        raise FileNotFoundError("images/ folder not found in the current directory.")

    output_dir = "results_TextRepresentation"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv("labels.csv")

    # Keep split style aligned with updated baseline.
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    transform = SimpleImageTransform(image_size=64)
    train_ds = SlotDataset(train_df, "images", transform=transform)
    test_ds = SlotDataset(test_df, "images", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlledSlotModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 15
    history_rows = []

    print(f"Training controlled slot model on {len(train_df)} train samples using {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        avg_loss = running_loss / max(batch_count, 1)
        history_rows.append({"epoch": epoch, "train_loss": avg_loss})
        print(f"Epoch {epoch:02d}/{epochs} - train_loss: {avg_loss:.4f}")

    metrics, details_df = evaluate_model(model, test_loader, device)
    metrics["split_sizes"] = {
        "train": int(len(train_df)),
        "test": int(len(test_df)),
    }
    metrics["model_notes"] = {
        "image_size": 64,
        "encoder": "updated_baseline_aligned_4layer_cnn",
        "comparison_purpose": "controlled_output_representation_comparison",
        "output_format": "structured_slot_based",
        "loss": "cross_entropy_per_slot",
        "split_alignment": "same style as updated baseline_CNN.py (train/test, test_size=0.15, random_state=42)",
        "random_seed": 42,
        "checkpoint_note": "final epoch checkpoint; no validation-based best-epoch selection",
    }

    pd.DataFrame(history_rows).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    details_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    rel1_conf = pd.crosstab(details_df["gold_rel1"], details_df["pred_rel1"])
    rel2_conf = pd.crosstab(details_df["gold_rel2"], details_df["pred_rel2"])
    rel1_conf.to_csv(os.path.join(output_dir, "test_rel1_confusion.csv"))
    rel2_conf.to_csv(os.path.join(output_dir, "test_rel2_confusion.csv"))

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    torch.save(model.state_dict(), os.path.join(output_dir, "best_slot_model_controlled.pt"))

    print("Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
