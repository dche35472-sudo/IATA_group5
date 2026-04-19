import os
import re
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


SEED = 42
IMAGE_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
REL_AXIS_WEIGHT = 2.0
REL_DIR_WEIGHT = 2.0
PATIENCE = 6
OUTPUT_DIR = "slot_model_outputs_v22"


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Label schema
# ----------------------------
SIZE_TO_ID = {"small": 0, "medium": 1, "big": 2}
COLOR_TO_ID = {"red": 0, "blue": 1, "green": 2, "yellow": 3, "black": 4}
SHAPE_TO_ID = {"circle": 0, "square": 1, "triangle": 2}
REL_TO_ID = {"above": 0, "below": 1, "left of": 2, "right of": 3, "overlapping": 4}
ID_TO_REL = {v: k for k, v in REL_TO_ID.items()}

# Factorised relation labels
AXIS_TO_ID = {"vertical": 0, "horizontal": 1, "overlap": 2}
DIR_TO_ID = {"positive": 0, "negative": 1, "none": 2}
ID_TO_AXIS = {v: k for k, v in AXIS_TO_ID.items()}
ID_TO_DIR = {v: k for k, v in DIR_TO_ID.items()}


def relation_to_axis_dir(rel: str) -> Tuple[int, int]:
    if rel == "above":
        return AXIS_TO_ID["vertical"], DIR_TO_ID["positive"]
    if rel == "below":
        return AXIS_TO_ID["vertical"], DIR_TO_ID["negative"]
    if rel == "right of":
        return AXIS_TO_ID["horizontal"], DIR_TO_ID["positive"]
    if rel == "left of":
        return AXIS_TO_ID["horizontal"], DIR_TO_ID["negative"]
    return AXIS_TO_ID["overlap"], DIR_TO_ID["none"]


def axis_dir_to_relation(axis_id: int, dir_id: int) -> str:
    axis = ID_TO_AXIS[int(axis_id)]
    direction = ID_TO_DIR[int(dir_id)]
    if axis == "overlap":
        return "overlapping"
    if axis == "vertical":
        return "above" if direction == "positive" else "below"
    return "right of" if direction == "positive" else "left of"


DESC_RE = re.compile(
    r"^a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"is (above|below|left of|right of|overlapping) a (small|medium|big) "
    r"(red|blue|green|yellow|black) (circle|square|triangle) \| "
    r"a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"is (above|below|left of|right of|overlapping) a (small|medium|big) "
    r"(red|blue|green|yellow|black) (circle|square|triangle)$"
)


@dataclass
class ParsedExample:
    file_name: str
    description: str
    anchor_size: int
    anchor_color: int
    anchor_shape: int
    rel1: int
    rel1_axis: int
    rel1_dir: int
    target1_size: int
    target1_color: int
    target1_shape: int
    rel2: int
    rel2_axis: int
    rel2_dir: int
    target2_size: int
    target2_color: int
    target2_shape: int


def parse_description(file_name: str, desc: str) -> ParsedExample:
    m = DESC_RE.match(desc.strip())
    if not m:
        raise ValueError(f"Cannot parse description: {desc}")

    (
        a_size, a_color, a_shape,
        rel1,
        t1_size, t1_color, t1_shape,
        a2_size, a2_color, a2_shape,
        rel2,
        t2_size, t2_color, t2_shape,
    ) = m.groups()

    # sanity check: anchor repeated in both clauses
    if (a_size, a_color, a_shape) != (a2_size, a2_color, a2_shape):
        raise ValueError(f"Anchor mismatch in description: {desc}")

    rel1_axis, rel1_dir = relation_to_axis_dir(rel1)
    rel2_axis, rel2_dir = relation_to_axis_dir(rel2)

    return ParsedExample(
        file_name=file_name,
        description=desc,
        anchor_size=SIZE_TO_ID[a_size],
        anchor_color=COLOR_TO_ID[a_color],
        anchor_shape=SHAPE_TO_ID[a_shape],
        rel1=REL_TO_ID[rel1],
        rel1_axis=rel1_axis,
        rel1_dir=rel1_dir,
        target1_size=SIZE_TO_ID[t1_size],
        target1_color=COLOR_TO_ID[t1_color],
        target1_shape=SHAPE_TO_ID[t1_shape],
        rel2=REL_TO_ID[rel2],
        rel2_axis=rel2_axis,
        rel2_dir=rel2_dir,
        target2_size=SIZE_TO_ID[t2_size],
        target2_color=COLOR_TO_ID[t2_color],
        target2_shape=SHAPE_TO_ID[t2_shape],
    )


# ----------------------------
# Dataset
# ----------------------------
class ShapeSentenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, image_size: int = IMAGE_SIZE):
        self.images_dir = images_dir
        self.image_size = image_size
        self.rows: List[ParsedExample] = []
        for _, row in df.iterrows():
            self.rows.append(parse_description(row["file_name"], row["description"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = os.path.join(self.images_dir, row.file_name)
        img = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.tensor(arr, dtype=torch.float32)

        labels = {
            "anchor_size": row.anchor_size,
            "anchor_color": row.anchor_color,
            "anchor_shape": row.anchor_shape,
            "rel1": row.rel1,
            "rel1_axis": row.rel1_axis,
            "rel1_dir": row.rel1_dir,
            "target1_size": row.target1_size,
            "target1_color": row.target1_color,
            "target1_shape": row.target1_shape,
            "rel2": row.rel2,
            "rel2_axis": row.rel2_axis,
            "rel2_dir": row.rel2_dir,
            "target2_size": row.target2_size,
            "target2_color": row.target2_color,
            "target2_shape": row.target2_shape,
            "description": row.description,
            "file_name": row.file_name,
        }
        return x, labels


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = {}
    for key in batch[0][1].keys():
        vals = [b[1][key] for b in batch]
        if key in {"description", "file_name"}:
            labels[key] = vals
        else:
            labels[key] = torch.tensor(vals, dtype=torch.long)
    return imgs, labels


# ----------------------------
# Model
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class DeeperCNNFactorized(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),   # 96 -> 48
            ConvBlock(32, 64),  # 48 -> 24
            ConvBlock(64, 128), # 24 -> 12
            ConvBlock(128, 256) # 12 -> 6
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # attribute heads
        self.anchor_size = nn.Linear(256, 3)
        self.anchor_color = nn.Linear(256, 5)
        self.anchor_shape = nn.Linear(256, 3)

        self.target1_size = nn.Linear(256, 3)
        self.target1_color = nn.Linear(256, 5)
        self.target1_shape = nn.Linear(256, 3)

        self.target2_size = nn.Linear(256, 3)
        self.target2_color = nn.Linear(256, 5)
        self.target2_shape = nn.Linear(256, 3)

        # factorized relation heads
        self.rel1_axis = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 3))
        self.rel1_dir = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 3))
        self.rel2_axis = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 3))
        self.rel2_dir = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 3))

    def forward(self, x):
        z = self.head(self.encoder(x))
        return {
            "anchor_size": self.anchor_size(z),
            "anchor_color": self.anchor_color(z),
            "anchor_shape": self.anchor_shape(z),
            "rel1_axis": self.rel1_axis(z),
            "rel1_dir": self.rel1_dir(z),
            "target1_size": self.target1_size(z),
            "target1_color": self.target1_color(z),
            "target1_shape": self.target1_shape(z),
            "rel2_axis": self.rel2_axis(z),
            "rel2_dir": self.rel2_dir(z),
            "target2_size": self.target2_size(z),
            "target2_color": self.target2_color(z),
            "target2_shape": self.target2_shape(z),
        }


# ----------------------------
# Utility functions
# ----------------------------
def decode_sentence(pred: Dict[str, int]) -> str:
    size_inv = {v: k for k, v in SIZE_TO_ID.items()}
    color_inv = {v: k for k, v in COLOR_TO_ID.items()}
    shape_inv = {v: k for k, v in SHAPE_TO_ID.items()}

    anchor = f"a {size_inv[pred['anchor_size']]} {color_inv[pred['anchor_color']]} {shape_inv[pred['anchor_shape']]}"
    target1 = f"a {size_inv[pred['target1_size']]} {color_inv[pred['target1_color']]} {shape_inv[pred['target1_shape']]}"
    target2 = f"a {size_inv[pred['target2_size']]} {color_inv[pred['target2_color']]} {shape_inv[pred['target2_shape']]}"
    rel1 = axis_dir_to_relation(pred["rel1_axis"], pred["rel1_dir"])
    rel2 = axis_dir_to_relation(pred["rel2_axis"], pred["rel2_dir"])
    return f"{anchor} is {rel1} {target1} | {anchor} is {rel2} {target2}"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    attr_keys = [
        "anchor_size", "anchor_color", "anchor_shape",
        "target1_size", "target1_color", "target1_shape",
        "target2_size", "target2_color", "target2_shape",
    ]
    head_keys = attr_keys + ["rel1_axis", "rel1_dir", "rel2_axis", "rel2_dir"]

    total = 0
    correct_by_head = {k: 0 for k in head_keys}
    exact_sentence = 0
    all_slots_joint = 0
    relation_correct = 0
    attribute_correct = 0
    pred_rows = []

    rel1_rows = []
    rel2_rows = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)

        preds = {k: torch.argmax(v, dim=1).cpu() for k, v in outputs.items()}
        bs = imgs.size(0)
        total += bs

        # head accuracies
        for k in head_keys:
            correct_by_head[k] += (preds[k] == labels[k]).sum().item()

        # derived relation accuracy
        pred_rel1 = [REL_TO_ID[axis_dir_to_relation(a, d)] for a, d in zip(preds["rel1_axis"].tolist(), preds["rel1_dir"].tolist())]
        pred_rel2 = [REL_TO_ID[axis_dir_to_relation(a, d)] for a, d in zip(preds["rel2_axis"].tolist(), preds["rel2_dir"].tolist())]
        gold_rel1 = labels["rel1"].tolist()
        gold_rel2 = labels["rel2"].tolist()
        relation_correct += sum(int(p == g) for p, g in zip(pred_rel1, gold_rel1))
        relation_correct += sum(int(p == g) for p, g in zip(pred_rel2, gold_rel2))

        # attribute accuracy
        attribute_correct += sum((preds[k] == labels[k]).sum().item() for k in attr_keys)

        for i in range(bs):
            slot_ok = True
            pred_dict = {}
            for k in attr_keys:
                pred_dict[k] = int(preds[k][i].item())
                if pred_dict[k] != int(labels[k][i].item()):
                    slot_ok = False
            pred_dict["rel1_axis"] = int(preds["rel1_axis"][i].item())
            pred_dict["rel1_dir"] = int(preds["rel1_dir"][i].item())
            pred_dict["rel2_axis"] = int(preds["rel2_axis"][i].item())
            pred_dict["rel2_dir"] = int(preds["rel2_dir"][i].item())

            if pred_rel1[i] != gold_rel1[i] or pred_rel2[i] != gold_rel2[i]:
                slot_ok = False

            pred_sentence = decode_sentence(pred_dict)
            gold_sentence = labels["description"][i]
            if pred_sentence == gold_sentence:
                exact_sentence += 1
            if slot_ok:
                all_slots_joint += 1

            pred_rows.append({
                "file_name": labels["file_name"][i],
                "gold_sentence": gold_sentence,
                "pred_sentence": pred_sentence,
                "gold_rel1": ID_TO_REL[gold_rel1[i]],
                "pred_rel1": ID_TO_REL[pred_rel1[i]],
                "gold_rel2": ID_TO_REL[gold_rel2[i]],
                "pred_rel2": ID_TO_REL[pred_rel2[i]],
                **{f"gold_{k}": int(labels[k][i].item()) for k in attr_keys},
                **{f"pred_{k}": int(pred_dict[k]) for k in attr_keys},
                "gold_rel1_axis": int(labels["rel1_axis"][i].item()),
                "pred_rel1_axis": pred_dict["rel1_axis"],
                "gold_rel1_dir": int(labels["rel1_dir"][i].item()),
                "pred_rel1_dir": pred_dict["rel1_dir"],
                "gold_rel2_axis": int(labels["rel2_axis"][i].item()),
                "pred_rel2_axis": pred_dict["rel2_axis"],
                "gold_rel2_dir": int(labels["rel2_dir"][i].item()),
                "pred_rel2_dir": pred_dict["rel2_dir"],
            })
            rel1_rows.append((ID_TO_REL[gold_rel1[i]], ID_TO_REL[pred_rel1[i]]))
            rel2_rows.append((ID_TO_REL[gold_rel2[i]], ID_TO_REL[pred_rel2[i]]))

    metrics = {
        "exact_sentence_accuracy": exact_sentence / total,
        "all_slots_joint_accuracy": all_slots_joint / total,
        "mean_slot_accuracy": np.mean([correct_by_head[k] / total for k in head_keys]),
        "relation_accuracy": relation_correct / (2 * total),
        "attribute_accuracy": attribute_correct / (len(attr_keys) * total),
    }
    for k in head_keys:
        metrics[f"acc_{k}"] = correct_by_head[k] / total

    pred_df = pd.DataFrame(pred_rows)
    rel1_cm = pd.crosstab(
        pd.Series([g for g, _ in rel1_rows], name="gold_rel1"),
        pd.Series([p for _, p in rel1_rows], name="pred_rel1")
    )
    rel2_cm = pd.crosstab(
        pd.Series([g for g, _ in rel2_rows], name="gold_rel2"),
        pd.Series([p for _, p in rel2_rows], name="pred_rel2")
    )
    return metrics, pred_df, rel1_cm, rel2_cm


# ----------------------------
# Training
# ----------------------------
def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_path = "labels.csv"
    images_dir = "images"
    if not os.path.exists(data_path):
        raise FileNotFoundError("labels.csv not found in current directory")
    if not os.path.exists(images_dir):
        raise FileNotFoundError("images directory not found in current directory")

    df = pd.read_csv(data_path)
    required_cols = {"file_name", "description"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"labels.csv must contain columns: {required_cols}")

    # deterministic split
    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    n_test = int(round(0.15 * n))
    n_val = int(round(0.085 * n))  # yields 1275 for n=15000
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_ds = ShapeSentenceDataset(train_df, images_dir, IMAGE_SIZE)
    val_ds = ShapeSentenceDataset(val_df, images_dir, IMAGE_SIZE)
    test_ds = ShapeSentenceDataset(test_df, images_dir, IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeeperCNNFactorized().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ce = nn.CrossEntropyLoss()

    best_score = -1e9
    best_state = None
    patience_left = PATIENCE
    history = []

    attr_keys = [
        "anchor_size", "anchor_color", "anchor_shape",
        "target1_size", "target1_color", "target1_shape",
        "target2_size", "target2_color", "target2_shape",
    ]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels_t = {k: v.to(device) for k, v in labels.items() if k not in {"description", "file_name"}}
            out = model(imgs)

            loss = 0.0
            for k in attr_keys:
                loss = loss + ce(out[k], labels_t[k])
            loss = loss + REL_AXIS_WEIGHT * ce(out["rel1_axis"], labels_t["rel1_axis"])
            loss = loss + REL_DIR_WEIGHT * ce(out["rel1_dir"], labels_t["rel1_dir"])
            loss = loss + REL_AXIS_WEIGHT * ce(out["rel2_axis"], labels_t["rel2_axis"])
            loss = loss + REL_DIR_WEIGHT * ce(out["rel2_dir"], labels_t["rel2_dir"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_ds)
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        score = (
            0.50 * val_metrics["relation_accuracy"] +
            0.30 * val_metrics["mean_slot_accuracy"] +
            0.15 * val_metrics["attribute_accuracy"] +
            0.05 * val_metrics["exact_sentence_accuracy"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_mean_slot={val_metrics['mean_slot_accuracy']:.4f} | "
            f"val_relation={val_metrics['relation_accuracy']:.4f} | score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a best model.")

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUTPUT_DIR, "best_slot_model_v22.pt"))

    val_metrics, val_pred, val_rel1_cm, val_rel2_cm = evaluate(model, val_loader, device)
    test_metrics, test_pred, test_rel1_cm, test_rel2_cm = evaluate(model, test_loader, device)

    payload = {
        "validation": val_metrics,
        "test": test_metrics,
        "split_sizes": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
        },
        "model_notes": {
            "image_size": IMAGE_SIZE,
            "encoder": "deeper_cnn_with_batchnorm_factorized_relations",
            "optimizer": "AdamW",
            "selection_metric": "0.50 relation + 0.30 mean_slot + 0.15 attribute + 0.05 exact",
            "relation_axis_loss_weight": REL_AXIS_WEIGHT,
            "relation_direction_loss_weight": REL_DIR_WEIGHT,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    val_pred.to_csv(os.path.join(OUTPUT_DIR, "val_predictions.csv"), index=False)
    test_pred.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    val_rel1_cm.to_csv(os.path.join(OUTPUT_DIR, "val_rel1_confusion.csv"))
    val_rel2_cm.to_csv(os.path.join(OUTPUT_DIR, "val_rel2_confusion.csv"))
    test_rel1_cm.to_csv(os.path.join(OUTPUT_DIR, "test_rel1_confusion.csv"))
    test_rel2_cm.to_csv(os.path.join(OUTPUT_DIR, "test_rel2_confusion.csv"))

    print("Saved outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
