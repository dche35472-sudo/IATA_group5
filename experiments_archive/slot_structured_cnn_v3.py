
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


# =========================================================
# Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Config
# =========================================================
LABELS_CSV = "labels.csv"
IMAGES_DIR = "images"
OUTPUT_DIR = "slot_model_outputs_v3"

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 35
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8

# Loss balance
COORD_LOSS_WEIGHT = 0.8
DIVERSITY_LOSS_WEIGHT = 0.02

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DESC_PATTERN = re.compile(
    r"^a (\w+) (\w+) (\w+) is (above|below|left of|right of|overlapping) "
    r"a (\w+) (\w+) (\w+) \| "
    r"a (\w+) (\w+) (\w+) is (above|below|left of|right of|overlapping) "
    r"a (\w+) (\w+) (\w+)$"
)

SIZE_TO_PIXEL = {"small": 25, "medium": 45, "big": 65}


# =========================================================
# Label vocabularies
# =========================================================
SIZE_VOCAB = ["small", "medium", "big"]
COLOR_VOCAB = ["red", "blue", "green", "yellow", "black"]
SHAPE_VOCAB = ["circle", "square", "triangle"]
REL_VOCAB = ["above", "below", "left of", "overlapping", "right of"]

SIZE2IDX = {v: i for i, v in enumerate(SIZE_VOCAB)}
COLOR2IDX = {v: i for i, v in enumerate(COLOR_VOCAB)}
SHAPE2IDX = {v: i for i, v in enumerate(SHAPE_VOCAB)}
REL2IDX = {v: i for i, v in enumerate(REL_VOCAB)}

IDX2SIZE = {i: v for v, i in SIZE2IDX.items()}
IDX2COLOR = {i: v for v, i in COLOR2IDX.items()}
IDX2SHAPE = {i: v for v, i in SHAPE2IDX.items()}
IDX2REL = {i: v for v, i in REL2IDX.items()}


# =========================================================
# Parsing helpers
# =========================================================
def parse_description(desc: str) -> Dict[str, str]:
    m = DESC_PATTERN.match(desc)
    if not m:
        raise ValueError(f"Could not parse description: {desc}")

    g = m.groups()
    return {
        "anchor_size": g[0],
        "anchor_color": g[1],
        "anchor_shape": g[2],
        "rel1": g[3],
        "target1_size": g[4],
        "target1_color": g[5],
        "target1_shape": g[6],
        "rel2": g[10],
        "target2_size": g[11],
        "target2_color": g[12],
        "target2_shape": g[13],
    }


def decode_sentence(parts: Dict[str, str]) -> str:
    return (
        f"a {parts['anchor_size']} {parts['anchor_color']} {parts['anchor_shape']} is {parts['rel1']} "
        f"a {parts['target1_size']} {parts['target1_color']} {parts['target1_shape']} | "
        f"a {parts['anchor_size']} {parts['anchor_color']} {parts['anchor_shape']} is {parts['rel2']} "
        f"a {parts['target2_size']} {parts['target2_color']} {parts['target2_shape']}"
    )


def load_objects(json_text: str) -> List[Dict]:
    objs = json.loads(json_text)
    if len(objs) < 3:
        raise ValueError("Each row must contain at least 3 objects.")
    return objs


def infer_relation_from_geometry(
    anchor_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    anchor_size_label: str,
    target_size_label: str,
    image_size: int = IMAGE_SIZE,
) -> str:
    """
    Approximate generator rule from predicted coordinates and predicted size classes.
    This is not used as the main supervision target, but it can be useful for analysis
    or as a fallback if needed.
    """
    ax, ay = anchor_xy
    tx, ty = target_xy

    # convert normalized coords to pixel coordinates in resized image
    ax *= (image_size - 1)
    ay *= (image_size - 1)
    tx *= (image_size - 1)
    ty *= (image_size - 1)

    a_half = SIZE_TO_PIXEL[anchor_size_label] / 2.0 * (image_size / 224.0)
    t_half = SIZE_TO_PIXEL[target_size_label] / 2.0 * (image_size / 224.0)

    overlap = (abs(tx - ax) <= (a_half + t_half)) and (abs(ty - ay) <= (a_half + t_half))
    if overlap:
        return "overlapping"

    dx = tx - ax
    dy = ty - ay

    if abs(dx) >= abs(dy):
        return "left of" if dx > 0 else "right of"
    else:
        return "above" if dy > 0 else "below"


# =========================================================
# Dataset
# =========================================================
class SlotObjectDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = IMAGE_SIZE):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size

        # parse descriptions once
        parsed_list = [parse_description(x) for x in self.df["description"].tolist()]
        parsed_df = pd.DataFrame(parsed_list)
        self.df = pd.concat([self.df, parsed_df], axis=1)

        # parse object JSON once
        self.df["objects_parsed"] = self.df["objects"].apply(load_objects)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = os.path.join(IMAGES_DIR, row["file_name"])

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        image = torch.tensor(arr, dtype=torch.float32)

        objs = row["objects_parsed"]
        # Important dataset fact:
        # objects[0], objects[1], objects[2] correspond exactly to
        # anchor, target1, target2 in the description.
        coord_targets = np.array([
            [objs[0]["x"] / max(1, orig_w - 1), objs[0]["y"] / max(1, orig_h - 1)],
            [objs[1]["x"] / max(1, orig_w - 1), objs[1]["y"] / max(1, orig_h - 1)],
            [objs[2]["x"] / max(1, orig_w - 1), objs[2]["y"] / max(1, orig_h - 1)],
        ], dtype=np.float32)

        slots = {
            "anchor_size": SIZE2IDX[row["anchor_size"]],
            "anchor_color": COLOR2IDX[row["anchor_color"]],
            "anchor_shape": SHAPE2IDX[row["anchor_shape"]],
            "rel1": REL2IDX[row["rel1"]],
            "target1_size": SIZE2IDX[row["target1_size"]],
            "target1_color": COLOR2IDX[row["target1_color"]],
            "target1_shape": SHAPE2IDX[row["target1_shape"]],
            "rel2": REL2IDX[row["rel2"]],
            "target2_size": SIZE2IDX[row["target2_size"]],
            "target2_color": COLOR2IDX[row["target2_color"]],
            "target2_shape": SHAPE2IDX[row["target2_shape"]],
        }

        sample = {
            "image": image,
            "coords": torch.tensor(coord_targets, dtype=torch.float32),
            "file_name": row["file_name"],
            "ground_truth": row["description"],
        }
        for k, v in slots.items():
            sample[k] = torch.tensor(v, dtype=torch.long)
        return sample


# =========================================================
# Train / validation / test split
# =========================================================
def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = np.arange(len(df))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(0.765 * n_total)   # 11475 / 15000
    n_val = int(0.085 * n_total)     # 1275 / 15000
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, val_df, test_df


# =========================================================
# Model
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionObjectSlotModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 32, pool=True),    # 128 -> 64
            ConvBlock(32, 64, pool=True),   # 64 -> 32
            ConvBlock(64, 128, pool=True),  # 32 -> 16
            ConvBlock(128, 256, pool=True), # 16 -> 8
        )

        self.attn_head = nn.Conv2d(256, 3, kernel_size=1)

        self.obj_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.size_head = nn.Linear(256, len(SIZE_VOCAB))
        self.color_head = nn.Linear(256, len(COLOR_VOCAB))
        self.shape_head = nn.Linear(256, len(SHAPE_VOCAB))

        self.relation_mlp = nn.Sequential(
            nn.Linear(256 * 2 + 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, len(REL_VOCAB)),
        )

    def spatial_attention(self, feat: torch.Tensor):
        """
        feat: [B, C, H, W]
        Returns:
            attn_maps: [B, 3, H, W]
            coords: [B, 3, 2] in [0, 1]
            obj_feats: [B, 3, C]
        """
        b, c, h, w = feat.shape
        logits = self.attn_head(feat)              # [B, 3, H, W]
        attn = F.softmax(logits.view(b, 3, -1), dim=-1).view(b, 3, h, w)

        grid_y = torch.linspace(0.0, 1.0, h, device=feat.device).view(1, 1, h, 1)
        grid_x = torch.linspace(0.0, 1.0, w, device=feat.device).view(1, 1, 1, w)

        x = (attn * grid_x).sum(dim=(2, 3))
        y = (attn * grid_y).sum(dim=(2, 3))
        coords = torch.stack([x, y], dim=-1)      # [B, 3, 2]

        # weighted sum over feature map
        obj_feats = torch.einsum("bkhw,bchw->bkc", attn, feat)
        return attn, coords, obj_feats

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        attn_maps, coords, obj_feats = self.spatial_attention(feat)

        obj_h = self.obj_proj(obj_feats)  # [B, 3, 256]
        anchor_h, t1_h, t2_h = obj_h[:, 0], obj_h[:, 1], obj_h[:, 2]
        anchor_xy, t1_xy, t2_xy = coords[:, 0], coords[:, 1], coords[:, 2]

        # object attribute logits, shared heads across roles
        out = {
            "anchor_size": self.size_head(anchor_h),
            "anchor_color": self.color_head(anchor_h),
            "anchor_shape": self.shape_head(anchor_h),
            "target1_size": self.size_head(t1_h),
            "target1_color": self.color_head(t1_h),
            "target1_shape": self.shape_head(t1_h),
            "target2_size": self.size_head(t2_h),
            "target2_color": self.color_head(t2_h),
            "target2_shape": self.shape_head(t2_h),
        }

        # relation heads use anchor/target embeddings + coordinates
        pair1 = torch.cat([anchor_h, t1_h, anchor_xy, t1_xy, t1_xy - anchor_xy], dim=-1)
        pair2 = torch.cat([anchor_h, t2_h, anchor_xy, t2_xy, t2_xy - anchor_xy], dim=-1)

        out["rel1"] = self.relation_mlp(pair1)
        out["rel2"] = self.relation_mlp(pair2)

        out["coords"] = coords
        out["attn_maps"] = attn_maps
        return out


# =========================================================
# Metrics and evaluation
# =========================================================
SLOT_NAMES = [
    "anchor_size", "anchor_color", "anchor_shape",
    "rel1",
    "target1_size", "target1_color", "target1_shape",
    "rel2",
    "target2_size", "target2_color", "target2_shape",
]

ATTR_SLOT_NAMES = [
    "anchor_size", "anchor_color", "anchor_shape",
    "target1_size", "target1_color", "target1_shape",
    "target2_size", "target2_color", "target2_shape",
]

REL_SLOT_NAMES = ["rel1", "rel2"]


def batch_metrics_from_logits(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
    pred = {k: outputs[k].argmax(dim=1) for k in SLOT_NAMES}
    correct = {k: (pred[k] == batch[k]).detach().cpu().numpy().astype(int) for k in SLOT_NAMES}
    return pred, correct


def logits_to_sentence(outputs: Dict[str, torch.Tensor], i: int) -> str:
    pred_parts = {
        "anchor_size": IDX2SIZE[int(outputs["anchor_size"].argmax(dim=1)[i].item())],
        "anchor_color": IDX2COLOR[int(outputs["anchor_color"].argmax(dim=1)[i].item())],
        "anchor_shape": IDX2SHAPE[int(outputs["anchor_shape"].argmax(dim=1)[i].item())],
        "rel1": IDX2REL[int(outputs["rel1"].argmax(dim=1)[i].item())],
        "target1_size": IDX2SIZE[int(outputs["target1_size"].argmax(dim=1)[i].item())],
        "target1_color": IDX2COLOR[int(outputs["target1_color"].argmax(dim=1)[i].item())],
        "target1_shape": IDX2SHAPE[int(outputs["target1_shape"].argmax(dim=1)[i].item())],
        "rel2": IDX2REL[int(outputs["rel2"].argmax(dim=1)[i].item())],
        "target2_size": IDX2SIZE[int(outputs["target2_size"].argmax(dim=1)[i].item())],
        "target2_color": IDX2COLOR[int(outputs["target2_color"].argmax(dim=1)[i].item())],
        "target2_shape": IDX2SHAPE[int(outputs["target2_shape"].argmax(dim=1)[i].item())],
    }
    return decode_sentence(pred_parts)


def evaluate(model: nn.Module, loader: DataLoader, split_name: str):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k != "image":
                    batch_gpu[k] = v.to(DEVICE)

            outputs = model(images)
            pred, correct = batch_metrics_from_logits(batch_gpu, outputs)

            coords_pred = outputs["coords"].cpu().numpy()
            coords_gold = batch["coords"].cpu().numpy()

            bs = images.size(0)
            for i in range(bs):
                row = {
                    "split": split_name,
                    "file_name": batch["file_name"][i],
                    "ground_truth": batch["ground_truth"][i],
                    "predicted_sentence": logits_to_sentence(outputs, i),
                }

                exact_match = int(row["predicted_sentence"] == row["ground_truth"])
                row["exact_match"] = exact_match

                for slot in SLOT_NAMES:
                    gold_idx = int(batch[slot][i].item())
                    pred_idx = int(pred[slot][i].detach().cpu().numpy())
                    row[f"gold_{slot}"] = (
                        IDX2SIZE[gold_idx] if "size" in slot else
                        IDX2COLOR[gold_idx] if "color" in slot else
                        IDX2SHAPE[gold_idx] if "shape" in slot else
                        IDX2REL[gold_idx]
                    )
                    row[f"pred_{slot}"] = (
                        IDX2SIZE[pred_idx] if "size" in slot else
                        IDX2COLOR[pred_idx] if "color" in slot else
                        IDX2SHAPE[pred_idx] if "shape" in slot else
                        IDX2REL[pred_idx]
                    )
                    row[f"{slot}_correct"] = int(correct[slot][i])

                # auxiliary localization outputs
                role_names = ["anchor", "target1", "target2"]
                for j, role in enumerate(role_names):
                    row[f"gold_{role}_x"] = float(coords_gold[i, j, 0])
                    row[f"gold_{role}_y"] = float(coords_gold[i, j, 1])
                    row[f"pred_{role}_x"] = float(coords_pred[i, j, 0])
                    row[f"pred_{role}_y"] = float(coords_pred[i, j, 1])

                rows.append(row)

    pred_df = pd.DataFrame(rows)

    metrics = {
        "exact_sentence_accuracy": float(pred_df["exact_match"].mean()),
        "all_slots_joint_accuracy": float(pred_df[[f"{s}_correct" for s in SLOT_NAMES]].all(axis=1).mean()),
        "mean_slot_accuracy": float(np.mean([pred_df[f"{s}_correct"].mean() for s in SLOT_NAMES])),
        "relation_accuracy": float(np.mean([pred_df[f"{s}_correct"].mean() for s in REL_SLOT_NAMES])),
        "attribute_accuracy": float(np.mean([pred_df[f"{s}_correct"].mean() for s in ATTR_SLOT_NAMES])),
    }
    for s in SLOT_NAMES:
        metrics[f"acc_{s}"] = float(pred_df[f"{s}_correct"].mean())

    return metrics, pred_df


def save_confusions(df: pd.DataFrame, output_dir: str, prefix: str):
    rel1_cm = pd.crosstab(df["gold_rel1"], df["pred_rel1"])
    rel2_cm = pd.crosstab(df["gold_rel2"], df["pred_rel2"])
    rel1_cm.to_csv(os.path.join(output_dir, f"{prefix}_rel1_confusion.csv"))
    rel2_cm.to_csv(os.path.join(output_dir, f"{prefix}_rel2_confusion.csv"))


# =========================================================
# Loss
# =========================================================
def attention_diversity_loss(attn_maps: torch.Tensor) -> torch.Tensor:
    """
    Encourage attention heads not to collapse to exactly the same region.
    Kept weak because some objects may overlap.
    """
    b, k, h, w = attn_maps.shape
    loss = 0.0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            loss = loss + (attn_maps[:, i] * attn_maps[:, j]).mean()
            count += 1
    return loss / max(1, count)


def compute_loss(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
    total_ce = 0.0
    ce_by_slot = {}

    for slot in SLOT_NAMES:
        ce = F.cross_entropy(outputs[slot], batch[slot])
        total_ce = total_ce + ce
        ce_by_slot[slot] = float(ce.detach().cpu().item())

    coord_loss = F.smooth_l1_loss(outputs["coords"], batch["coords"])
    div_loss = attention_diversity_loss(outputs["attn_maps"])

    total_loss = total_ce + COORD_LOSS_WEIGHT * coord_loss + DIVERSITY_LOSS_WEIGHT * div_loss
    aux = {
        "coord_loss": float(coord_loss.detach().cpu().item()),
        "diversity_loss": float(div_loss.detach().cpu().item()),
    }
    aux.update(ce_by_slot)
    return total_loss, aux


# =========================================================
# Training
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(LABELS_CSV)
    train_df, val_df, test_df = split_dataframe(df)

    train_ds = SlotObjectDataset(train_df, image_size=IMAGE_SIZE)
    val_ds = SlotObjectDataset(val_df, image_size=IMAGE_SIZE)
    test_ds = SlotObjectDataset(test_df, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AttentionObjectSlotModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_score = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    history_rows = []

    print(f"Training on {len(train_ds)} samples; validating on {len(val_ds)}; testing on {len(test_ds)}.")
    print(f"Using device: {DEVICE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        batch_losses = []

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k != "image":
                    batch_gpu[k] = v.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_parts = compute_loss(batch_gpu, outputs)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses))

        val_metrics, _ = evaluate(model, val_loader, "validation")
        selection_score = (
            0.45 * val_metrics["mean_slot_accuracy"]
            + 0.25 * val_metrics["relation_accuracy"]
            + 0.20 * val_metrics["attribute_accuracy"]
            + 0.10 * val_metrics["exact_sentence_accuracy"]
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_exact_sentence_accuracy": val_metrics["exact_sentence_accuracy"],
            "val_all_slots_joint_accuracy": val_metrics["all_slots_joint_accuracy"],
            "val_mean_slot_accuracy": val_metrics["mean_slot_accuracy"],
            "val_relation_accuracy": val_metrics["relation_accuracy"],
            "val_attribute_accuracy": val_metrics["attribute_accuracy"],
            "selection_score": selection_score,
        }
        history_rows.append(history_row)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_exact={val_metrics['exact_sentence_accuracy']:.4f} | "
            f"val_joint={val_metrics['all_slots_joint_accuracy']:.4f} | "
            f"val_mean_slot={val_metrics['mean_slot_accuracy']:.4f} | "
            f"val_rel={val_metrics['relation_accuracy']:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No best model was saved.")

    model.load_state_dict(best_state)
    model.to(DEVICE)

    val_metrics, val_pred_df = evaluate(model, val_loader, "validation")
    test_metrics, test_pred_df = evaluate(model, test_loader, "test")

    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "split_sizes": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
        },
        "model_notes": {
            "image_size": IMAGE_SIZE,
            "encoder": "attention_object_slot_cnn_with_coord_supervision",
            "optimizer": "AdamW",
            "selection_metric": "0.45 mean_slot + 0.25 relation + 0.20 attribute + 0.10 exact",
            "coord_loss_weight": COORD_LOSS_WEIGHT,
            "diversity_loss_weight": DIVERSITY_LOSS_WEIGHT,
            "best_epoch": best_epoch,
        },
    }

    # save artifacts
    best_model_path = os.path.join(OUTPUT_DIR, "best_slot_model_v3.pt")
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    val_pred_path = os.path.join(OUTPUT_DIR, "val_predictions.csv")
    test_pred_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")

    torch.save(model.state_dict(), best_model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    val_pred_df.to_csv(val_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)

    save_confusions(val_pred_df, OUTPUT_DIR, "val")
    save_confusions(test_pred_df, OUTPUT_DIR, "test")

    print("\nBest validation selection score:", round(best_score, 4))
    print("Saved:")
    print("-", best_model_path)
    print("-", metrics_path)
    print("-", history_path)
    print("-", val_pred_path)
    print("-", test_pred_path)
    print("-", os.path.join(OUTPUT_DIR, "test_rel1_confusion.csv"))
    print("-", os.path.join(OUTPUT_DIR, "test_rel2_confusion.csv"))


if __name__ == "__main__":
    main()
