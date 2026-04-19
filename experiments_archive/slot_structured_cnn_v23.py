
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


SIZES = ["small", "medium", "big"]
COLORS = ["red", "blue", "green", "yellow", "black"]
SHAPES = ["circle", "square", "triangle"]
RELATIONS = ["left of", "right of", "above", "below", "overlapping"]
REL_AXES = ["horizontal", "vertical", "overlap"]
REL_DIRS = ["negative", "positive", "none"]

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
HEAD_TO_INDEX = {name: {label: i for i, label in enumerate(vocab)} for name, vocab in HEAD_SPECS}
AUX_HEAD_SPECS = [
    ("rel1_axis", REL_AXES),
    ("rel1_dir", REL_DIRS),
    ("rel2_axis", REL_AXES),
    ("rel2_dir", REL_DIRS),
]
AUX_HEAD_TO_INDEX = {name: {label: i for i, label in enumerate(vocab)} for name, vocab in AUX_HEAD_SPECS}

DESC_PATTERN = re.compile(
    r"^a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"is (left of|right of|above|below|overlapping) "
    r"a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"\| a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle) "
    r"is (left of|right of|above|below|overlapping) "
    r"a (small|medium|big) (red|blue|green|yellow|black) (circle|square|triangle)$"
)


def relation_to_axis_dir(relation: str) -> Tuple[str, str]:
    if relation == "left of":
        return "horizontal", "negative"
    if relation == "right of":
        return "horizontal", "positive"
    if relation == "above":
        return "vertical", "negative"
    if relation == "below":
        return "vertical", "positive"
    if relation == "overlapping":
        return "overlap", "none"
    raise ValueError(f"Unknown relation: {relation}")


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

    rel1_axis, rel1_dir = relation_to_axis_dir(rel1)
    rel2_axis, rel2_dir = relation_to_axis_dir(rel2)

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
        "rel1_axis": rel1_axis,
        "rel1_dir": rel1_dir,
        "rel2_axis": rel2_axis,
        "rel2_dir": rel2_dir,
    }


def encode_slots(slot_dict: Dict[str, str]) -> Dict[str, int]:
    encoded = {head: HEAD_TO_INDEX[head][slot_dict[head]] for head in HEAD_NAMES}
    encoded.update({head: AUX_HEAD_TO_INDEX[head][slot_dict[head]] for head, _ in AUX_HEAD_SPECS})
    return encoded


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


class SlotDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform or SimpleImageTransform()

        parsed = self.df["description"].apply(parse_description)
        encoded = parsed.apply(encode_slots)

        for head in HEAD_NAMES + [name for name, _ in AUX_HEAD_SPECS]:
            self.df[head] = encoded.apply(lambda x: x[head])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["file_name"]))
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        target = {head: torch.tensor(int(row[head]), dtype=torch.long) for head in HEAD_NAMES}
        aux_target = {head: torch.tensor(int(row[head]), dtype=torch.long) for head, _ in AUX_HEAD_SPECS}
        meta = {"file_name": row["file_name"], "description": row["description"]}
        return image, target, aux_target, meta


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
            ConvBNAct(3, 32), ConvBNAct(32, 32), nn.MaxPool2d(2),
            ConvBNAct(32, 64), ConvBNAct(64, 64), nn.MaxPool2d(2),
            ConvBNAct(64, 128), ConvBNAct(128, 128), nn.MaxPool2d(2),
            ConvBNAct(128, 256), ConvBNAct(256, 256), nn.MaxPool2d(2),
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class SlotPredictionCNNV23(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeeperCNNEncoder()
        self.shared = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({head: nn.Linear(512, len(vocab)) for head, vocab in HEAD_SPECS})
        self.rel_aux = nn.ModuleDict({head: nn.Linear(512, len(vocab)) for head, vocab in AUX_HEAD_SPECS})

    def forward(self, x: torch.Tensor):
        z = self.shared(self.encoder(x))
        outputs = {head: classifier(z) for head, classifier in self.heads.items()}
        aux_outputs = {head: classifier(z) for head, classifier in self.rel_aux.items()}
        return outputs, aux_outputs


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
    aux_targets = {head: torch.stack([item[2][head] for item in batch]) for head, _ in AUX_HEAD_SPECS}
    metas = [item[3] for item in batch]
    return images, targets, aux_targets, metas


def compute_loss(outputs, aux_outputs, targets, aux_targets, criterions, aux_criterions):
    main_losses = [criterions[head](outputs[head], targets[head]) for head in HEAD_NAMES]
    aux_losses = [aux_criterions[head](aux_outputs[head], aux_targets[head]) for head, _ in AUX_HEAD_SPECS]
    main_loss = torch.stack(main_losses).mean()
    aux_loss = torch.stack(aux_losses).mean()
    return main_loss + 0.30 * aux_loss


def predictions_to_index_dict(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {head: logits.argmax(dim=1) for head, logits in outputs.items()}


def evaluate_model(model, data_loader, device, split_name="test"):
    model.eval()
    correct_per_head = {head: 0 for head in HEAD_NAMES}
    total = 0
    joint_correct = 0
    exact_sentence_correct = 0
    prediction_rows = []

    with torch.no_grad():
        for images, targets, aux_targets, metas in data_loader:
            images = images.to(device)
            targets = {head: value.to(device) for head, value in targets.items()}
            outputs, aux_outputs = model(images)
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
    relation_heads = ["rel1", "rel2"]
    attribute_heads = [head for head in HEAD_NAMES if head not in relation_heads]
    metrics = Metrics(
        exact_sentence_accuracy=exact_sentence_correct / total,
        all_slots_joint_accuracy=joint_correct / total,
        mean_slot_accuracy=float(np.mean(list(per_head_accuracy.values()))),
        relation_accuracy=float(np.mean([per_head_accuracy[h] for h in relation_heads])),
        attribute_accuracy=float(np.mean([per_head_accuracy[h] for h in attribute_heads])),
        per_head_accuracy=per_head_accuracy,
    )
    return metrics, pd.DataFrame(prediction_rows)


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
    return 0.6 * metrics.mean_slot_accuracy + 0.4 * metrics.relation_accuracy


def save_relation_confusions(pred_df: pd.DataFrame, output_dir: str, prefix: str) -> None:
    pd.crosstab(pred_df["gold_rel1"], pred_df["pred_rel1"]).to_csv(os.path.join(output_dir, f"{prefix}_rel1_confusion.csv"))
    pd.crosstab(pred_df["gold_rel2"], pred_df["pred_rel2"]).to_csv(os.path.join(output_dir, f"{prefix}_rel2_confusion.csv"))


def main():
    set_seed(42)
    labels_path = "labels.csv"
    images_dir = "images"
    output_dir = "slot_model_outputs_v23"
    if not os.path.exists(labels_path):
        raise FileNotFoundError("labels.csv not found in the current working directory.")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError("images/ directory not found in the current working directory.")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(labels_path)
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
    model = SlotPredictionCNNV23().to(device)

    criterions = {head: nn.CrossEntropyLoss(label_smoothing=0.02) for head in HEAD_NAMES}
    aux_criterions = {head: nn.CrossEntropyLoss(label_smoothing=0.02) for head, _ in AUX_HEAD_SPECS}
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    epochs = 25
    early_stop_patience = 6
    history = []
    best_score = -1.0
    best_model_path = os.path.join(output_dir, "best_slot_model_v23.pt")
    patience_counter = 0

    print(f"Training on {len(train_df)} samples; validating on {len(val_df)}; testing on {len(test_df)}.")
    print(f"Using device: {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets, aux_targets, _ in train_loader:
            images = images.to(device)
            targets = {head: value.to(device) for head, value in targets.items()}
            aux_targets = {head: value.to(device) for head, value in aux_targets.items()}
            optimizer.zero_grad()
            outputs, aux_outputs = model(images)
            loss = compute_loss(outputs, aux_outputs, targets, aux_targets, criterions, aux_criterions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_metrics, _ = evaluate_model(model, val_loader, device, split_name="val")
        selection_score = compute_selection_score(val_metrics)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_exact_sentence_accuracy": val_metrics.exact_sentence_accuracy,
            "val_all_slots_joint_accuracy": val_metrics.all_slots_joint_accuracy,
            "val_mean_slot_accuracy": val_metrics.mean_slot_accuracy,
            "val_relation_accuracy": val_metrics.relation_accuracy,
            "selection_score": selection_score,
            "learning_rate": current_lr,
        })
        print(
            f"Epoch {epoch:02d} | lr={current_lr:.6f} | train_loss={train_loss:.4f} | "
            f"val_exact={val_metrics.exact_sentence_accuracy:.4f} | "
            f"val_joint={val_metrics.all_slots_joint_accuracy:.4f} | "
            f"val_mean_slot={val_metrics.mean_slot_accuracy:.4f} | "
            f"val_relation={val_metrics.relation_accuracy:.4f} | score={selection_score:.4f}"
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
        "split_sizes": {"train": int(len(train_df)), "validation": int(len(val_df)), "test": int(len(test_df))},
        "model_notes": {
            "image_size": 96,
            "encoder": "deeper_cnn_with_batchnorm_plus_aux_relation_factorization",
            "optimizer": "AdamW",
            "selection_metric": "0.6 * val_mean_slot_accuracy + 0.4 * val_relation_accuracy",
            "aux_relation_loss_weight": 0.30,
        },
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    val_preds.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)
    test_preds.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    save_relation_confusions(val_preds, output_dir, prefix="val")
    save_relation_confusions(test_preds, output_dir, prefix="test")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
