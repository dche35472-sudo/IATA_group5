import os
import re
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. Image transform
# IMPORTANT: no flips/rotations because they would corrupt
# left/right and above/below labels.
# =========================================================
class SimpleImageTransform:
    def __init__(self, image_size: int = 96):
        self.image_size = image_size
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr, dtype=torch.float32)


# =========================================================
# 3. Label schema
# =========================================================
SIZES = ["small", "medium", "big"]
COLORS = ["red", "blue", "green", "yellow", "black"]
SHAPES = ["circle", "square", "triangle"]
RELATIONS = ["left of", "right of", "above", "below", "overlapping"]

HEAD_SPECS: List[Tuple[str, List[str]]] = [
    ("anchor_size", SIZES),
    ("anchor_color", COLORS),
    ("anchor_shape", SHAPES),
    ("rel1", RELATIONS),
    ("target1_size", SIZES),
    ("target1_color", COLORS),
    ("target1_shape", SHAPES),
    ("rel2", RELATIONS),
    ("target2_size", SIZES),
    ("target2_color", COLORS),
    ("target2_shape", SHAPES),
]

HEAD_NAMES = [name for name, _ in HEAD_SPECS]
RELATION_HEADS = ["rel1", "rel2"]
ATTRIBUTE_HEADS = [head for head in HEAD_NAMES if head not in RELATION_HEADS]
HEAD_TO_INDEX = {name: {label: i for i, label in enumerate(vocab)} for name, vocab in HEAD_SPECS}

DESC_PATTERN = re.compile(
    r"^a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"is (left of|right of|above|below|overlapping) "
    r"a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"\| a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"is (left of|right of|above|below|overlapping) "
    r"a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle)$"
)


def parse_description(description: str) -> Dict[str, str]:
    match = DESC_PATTERN.match(description.strip().lower())
    if not match:
        raise ValueError(f"Description does not match expected template:\n{description}")

    (
        anchor_size_1,
        anchor_color_1,
        anchor_shape_1,
        rel1,
        target1_size,
        target1_color,
        target1_shape,
        anchor_size_2,
        anchor_color_2,
        anchor_shape_2,
        rel2,
        target2_size,
        target2_color,
        target2_shape,
    ) = match.groups()

    if (anchor_size_1, anchor_color_1, anchor_shape_1) != (anchor_size_2, anchor_color_2, anchor_shape_2):
        raise ValueError(f"Anchor mismatch between clauses:\n{description}")

    return {
        "anchor_size": anchor_size_1,
        "anchor_color": anchor_color_1,
        "anchor_shape": anchor_shape_1,
        "rel1": rel1,
        "target1_size": target1_size,
        "target1_color": target1_color,
        "target1_shape": target1_shape,
        "rel2": rel2,
        "target2_size": target2_size,
        "target2_color": target2_color,
        "target2_shape": target2_shape,
    }


def encode_slots(slot_dict: Dict[str, str]) -> Dict[str, int]:
    return {head: HEAD_TO_INDEX[head][label] for head, label in slot_dict.items()}


def decode_slots(index_dict: Dict[str, int]) -> Dict[str, str]:
    decoded = {}
    for head, vocab in HEAD_SPECS:
        decoded[head] = vocab[index_dict[head]]
    return decoded


def slot_dict_to_sentence(slot_dict: Dict[str, str]) -> str:
    anchor = f"{slot_dict['anchor_size']} {slot_dict['anchor_color']} {slot_dict['anchor_shape']}"
    target1 = f"{slot_dict['target1_size']} {slot_dict['target1_color']} {slot_dict['target1_shape']}"
    target2 = f"{slot_dict['target2_size']} {slot_dict['target2_color']} {slot_dict['target2_shape']}"
    return (
        f"a {anchor} is {slot_dict['rel1']} a {target1} | "
        f"a {anchor} is {slot_dict['rel2']} a {target2}"
    )


# =========================================================
# 4. Dataset
# =========================================================
class SlotDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform or SimpleImageTransform()

        parsed = self.df["description"].apply(parse_description)
        encoded = parsed.apply(encode_slots)

        for head in HEAD_NAMES:
            self.df[head] = encoded.apply(lambda x: x[head])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["file_name"]))
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        target = {head: torch.tensor(int(row[head]), dtype=torch.long) for head in HEAD_NAMES}
        meta = {
            "file_name": row["file_name"],
            "description": row["description"],
        }
        return image, target, meta


# =========================================================
# 5. Model
# =========================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeeperCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32),
            ConvBNAct(32, 32),
            nn.MaxPool2d(2),

            ConvBNAct(32, 64),
            ConvBNAct(64, 64),
            nn.MaxPool2d(2),

            ConvBNAct(64, 128),
            ConvBNAct(128, 128),
            nn.MaxPool2d(2),

            ConvBNAct(128, 256),
            ConvBNAct(256, 256),
            nn.MaxPool2d(2),
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.projection(x)
        return x


class SlotPredictionCNNV21(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeeperCNNEncoder()

        self.attribute_heads = nn.ModuleDict(
            {head: nn.Linear(512, len(vocab)) for head, vocab in HEAD_SPECS if head in ATTRIBUTE_HEADS}
        )

        self.relation_heads = nn.ModuleDict(
            {
                head: nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.35),
                    nn.Linear(256, len(RELATIONS)),
                )
                for head in RELATION_HEADS
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        outputs = {head: classifier(z) for head, classifier in self.attribute_heads.items()}
        outputs.update({head: classifier(z) for head, classifier in self.relation_heads.items()})
        return outputs


# =========================================================
# 6. Metrics and utilities
# =========================================================
@dataclass
class Metrics:
    exact_sentence_accuracy: float
    all_slots_joint_accuracy: float
    mean_slot_accuracy: float
    relation_accuracy: float
    attribute_accuracy: float
    per_head_accuracy: Dict[str, float]


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch])
    targets = {head: torch.stack([item[1][head] for item in batch]) for head in HEAD_NAMES}
    metas = [item[2] for item in batch]
    return images, targets, metas


def build_relation_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    parsed = train_df["description"].apply(parse_description)
    rels = pd.concat([parsed.apply(lambda x: x["rel1"]), parsed.apply(lambda x: x["rel2"])], ignore_index=True)
    counts = rels.value_counts().reindex(RELATIONS).astype(float)

    # Mild weighting: inverse sqrt frequency, normalized to mean 1.0
    weights = np.sqrt(counts.sum() / (len(counts) * counts.values))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    criterions: Dict[str, nn.Module],
    head_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weighted_losses = []
    raw_loss_log = {}

    for head in HEAD_NAMES:
        loss_value = criterions[head](outputs[head], targets[head])
        raw_loss_log[f"loss_{head}"] = float(loss_value.item())
        weighted_losses.append(loss_value * head_weights[head])

    total_weight = float(sum(head_weights.values()))
    total_loss = torch.stack(weighted_losses).sum() / total_weight
    return total_loss, raw_loss_log


def predictions_to_index_dict(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {head: logits.argmax(dim=1) for head, logits in outputs.items()}


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
) -> Tuple[Metrics, pd.DataFrame]:
    model.eval()

    correct_per_head = {head: 0 for head in HEAD_NAMES}
    total = 0
    joint_correct = 0
    exact_sentence_correct = 0
    prediction_rows = []

    with torch.no_grad():
        for images, targets, metas in data_loader:
            images = images.to(device)
            targets = {head: value.to(device) for head, value in targets.items()}

            outputs = model(images)
            pred_indices = predictions_to_index_dict(outputs)

            batch_size = images.size(0)
            total += batch_size
            joint_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

            for head in HEAD_NAMES:
                head_correct = (pred_indices[head] == targets[head])
                correct_per_head[head] += int(head_correct.sum().item())
                joint_mask = joint_mask & head_correct

            joint_correct += int(joint_mask.sum().item())

            for i in range(batch_size):
                pred_index_dict = {head: int(pred_indices[head][i].item()) for head in HEAD_NAMES}
                gold_index_dict = {head: int(targets[head][i].item()) for head in HEAD_NAMES}

                pred_slot_dict = decode_slots(pred_index_dict)
                gold_slot_dict = decode_slots(gold_index_dict)

                pred_sentence = slot_dict_to_sentence(pred_slot_dict)
                gold_sentence = metas[i]["description"]
                exact_match = int(pred_sentence == gold_sentence)
                exact_sentence_correct += exact_match

                row = {
                    "split": split_name,
                    "file_name": metas[i]["file_name"],
                    "ground_truth": gold_sentence,
                    "predicted_sentence": pred_sentence,
                    "exact_match": exact_match,
                }
                for head in HEAD_NAMES:
                    row[f"gold_{head}"] = gold_slot_dict[head]
                    row[f"pred_{head}"] = pred_slot_dict[head]
                    row[f"{head}_correct"] = int(gold_slot_dict[head] == pred_slot_dict[head])

                prediction_rows.append(row)

    per_head_accuracy = {head: correct_per_head[head] / total for head in HEAD_NAMES}

    metrics = Metrics(
        exact_sentence_accuracy=exact_sentence_correct / total,
        all_slots_joint_accuracy=joint_correct / total,
        mean_slot_accuracy=float(np.mean(list(per_head_accuracy.values()))),
        relation_accuracy=float(np.mean([per_head_accuracy[h] for h in RELATION_HEADS])),
        attribute_accuracy=float(np.mean([per_head_accuracy[h] for h in ATTRIBUTE_HEADS])),
        per_head_accuracy=per_head_accuracy,
    )

    prediction_df = pd.DataFrame(prediction_rows)
    return metrics, prediction_df


def format_metrics(metrics: Metrics) -> Dict[str, float]:
    flat = {
        "exact_sentence_accuracy": metrics.exact_sentence_accuracy,
        "all_slots_joint_accuracy": metrics.all_slots_joint_accuracy,
        "mean_slot_accuracy": metrics.mean_slot_accuracy,
        "relation_accuracy": metrics.relation_accuracy,
        "attribute_accuracy": metrics.attribute_accuracy,
    }
    flat.update({f"acc_{head}": acc for head, acc in metrics.per_head_accuracy.items()})
    return flat


def compute_selection_score(metrics: Metrics) -> float:
    # Relation is the known bottleneck, so make it matter more in model selection.
    return (
        0.45 * metrics.relation_accuracy
        + 0.35 * metrics.mean_slot_accuracy
        + 0.15 * metrics.attribute_accuracy
        + 0.05 * metrics.exact_sentence_accuracy
    )


def save_relation_confusions(pred_df: pd.DataFrame, output_dir: str, prefix: str) -> None:
    rel1_cm = pd.crosstab(pred_df["gold_rel1"], pred_df["pred_rel1"])
    rel2_cm = pd.crosstab(pred_df["gold_rel2"], pred_df["pred_rel2"])
    rel1_cm.to_csv(os.path.join(output_dir, f"{prefix}_rel1_confusion.csv"))
    rel2_cm.to_csv(os.path.join(output_dir, f"{prefix}_rel2_confusion.csv"))


# =========================================================
# 7. Train
# =========================================================
def main():
    set_seed(42)

    labels_path = "labels.csv"
    images_dir = "images"
    output_dir = "slot_model_outputs_v21"

    if not os.path.exists(labels_path):
        raise FileNotFoundError("labels.csv not found in the current working directory.")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError("images/ directory not found in the current working directory.")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(labels_path)

    # Keep the same split logic as v1/v2/v3 for comparability.
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)
    train_df, val_df = train_test_split(train_val_df, test_size=0.10, random_state=42, shuffle=True)

    transform = SimpleImageTransform(image_size=96)
    train_dataset = SlotDataset(train_df, images_dir, transform=transform)
    val_dataset = SlotDataset(val_df, images_dir, transform=transform)
    test_dataset = SlotDataset(test_df, images_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlotPredictionCNNV21().to(device)

    relation_class_weights = build_relation_class_weights(train_df).to(device)

    criterions = {
        head: nn.CrossEntropyLoss(
            weight=relation_class_weights if head in RELATION_HEADS else None,
            label_smoothing=0.0 if head in RELATION_HEADS else 0.02,
        )
        for head in HEAD_NAMES
    }

    head_weights = {head: 1.0 for head in HEAD_NAMES}
    head_weights["rel1"] = 2.5
    head_weights["rel2"] = 2.5

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    epochs = 28
    early_stop_patience = 7
    history = []
    best_score = -1.0
    best_model_path = os.path.join(output_dir, "best_slot_model_v21.pt")
    patience_counter = 0

    print(f"Training on {len(train_df)} samples; validating on {len(val_df)}; testing on {len(test_df)}.")
    print(f"Using device: {device}")
    print(f"Relation class weights: {relation_class_weights.detach().cpu().numpy().round(4).tolist()}")
    print(f"Head weights: {head_weights}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loss_tracker = {f"loss_{head}": [] for head in HEAD_NAMES}

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = {head: value.to(device) for head, value in targets.items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss, raw_loss_log = compute_loss(outputs, targets, criterions, head_weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for key, value in raw_loss_log.items():
                loss_tracker[key].append(value)

        train_loss = running_loss / len(train_loader)
        mean_head_losses = {key: float(np.mean(values)) for key, values in loss_tracker.items()}
        val_metrics, _ = evaluate_model(model, val_loader, device, split_name="val")
        selection_score = compute_selection_score(val_metrics)
        current_lr = optimizer.param_groups[0]["lr"]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_exact_sentence_accuracy": val_metrics.exact_sentence_accuracy,
            "val_all_slots_joint_accuracy": val_metrics.all_slots_joint_accuracy,
            "val_mean_slot_accuracy": val_metrics.mean_slot_accuracy,
            "val_relation_accuracy": val_metrics.relation_accuracy,
            "val_attribute_accuracy": val_metrics.attribute_accuracy,
            "selection_score": selection_score,
            "learning_rate": current_lr,
        }
        history_row.update(mean_head_losses)
        history.append(history_row)

        print(
            f"Epoch {epoch:02d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_exact={val_metrics.exact_sentence_accuracy:.4f} | "
            f"val_mean_slot={val_metrics.mean_slot_accuracy:.4f} | "
            f"val_relation={val_metrics.relation_accuracy:.4f} | "
            f"val_attribute={val_metrics.attribute_accuracy:.4f} | "
            f"score={selection_score:.4f}"
        )

        scheduler.step(selection_score)

        if selection_score > best_score:
            best_score = selection_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch:02d}.")
                break

    print(f"Best validation selection score: {best_score:.4f}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_metrics, val_preds = evaluate_model(model, val_loader, device, split_name="val")
    test_metrics, test_preds = evaluate_model(model, test_loader, device, split_name="test")

    metrics_payload = {
        "validation": format_metrics(val_metrics),
        "test": format_metrics(test_metrics),
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "model_notes": {
            "image_size": 96,
            "encoder": "deeper_cnn_with_batchnorm_and_relation_emphasis",
            "optimizer": "AdamW",
            "selection_metric": "0.45 relation + 0.35 mean_slot + 0.15 attribute + 0.05 exact",
            "relation_head_loss_weight": 2.5,
            "relation_class_weights": relation_class_weights.detach().cpu().numpy().round(6).tolist(),
        },
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    val_preds.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)
    test_preds.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    save_relation_confusions(val_preds, output_dir, prefix="val")
    save_relation_confusions(test_preds, output_dir, prefix="test")

    print("Saved outputs to:", output_dir)


if __name__ == "__main__":
    main()
