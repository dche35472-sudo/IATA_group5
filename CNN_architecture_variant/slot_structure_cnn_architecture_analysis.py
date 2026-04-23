import os
import json
import re
import pandas as pd


LABELS_PATH = "labels.csv"
MODEL_FILES = {
    "SimpleCNN": "slot_simple_CNN_outputs/test_predictions.csv",
    "ResNet18": "slot_ResNet_outputs/test_predictions.csv",
    #"DeeperCNN": "slot_deeper_CNN_outputs/test_predictions.csv",   # optional
}


DESC_RE = re.compile(
    r"^a (\w+) (\w+) (\w+) is (left of|right of|above|below|overlapping) "
    r"a (\w+) (\w+) (\w+) \| "
    r"a (\w+) (\w+) (\w+) is (left of|right of|above|below|overlapping) "
    r"a (\w+) (\w+) (\w+)$"
)


def parse_description(text):
    text = str(text).strip().lower()
    m = DESC_RE.match(text)
    if not m:
        return None

    g = m.groups()
    return {
        "anchor_size": g[0],
        "anchor_color": g[1],
        "anchor_shape": g[2],
        "rel1": g[3],
        "target1_size": g[4],
        "target1_color": g[5],
        "target1_shape": g[6],
        "anchor2_size": g[7],
        "anchor2_color": g[8],
        "anchor2_shape": g[9],
        "rel2": g[10],
        "target2_size": g[11],
        "target2_color": g[12],
        "target2_shape": g[13],
    }


# complexity information from labels
# three complexity variable:
# object_count: the num of objects in the image
# has_overlapping: is there an overlapping relationship
# relation_pair_type: are the two relations the same
def build_complexity_table(labels_df):
    rows = []

    for _, row in labels_df.iterrows():
        file_name = row["file_name"]
        description = row["description"]
        objects = json.loads(row["objects"])

        parsed = parse_description(description)
        if parsed is None:
            continue

        object_count = len(objects)

        rel1 = parsed["rel1"]
        rel2 = parsed["rel2"]

        has_overlapping = int((rel1 == "overlapping") or (rel2 == "overlapping"))

        if rel1 == rel2:
            relation_pair_type = "same_relation"
        else:
            relation_pair_type = "mixed_relation"

        rows.append(
            {
                "file_name": file_name,
                "object_count": object_count,
                "has_overlapping": has_overlapping,
                "relation_pair_type": relation_pair_type,
                "rel1": rel1,
                "rel2": rel2,
                "ground_truth": description,
            }
        )

    return pd.DataFrame(rows)


# Calculate the various accuracy metrics for each sample
def add_row_metrics(pred_df):
    correct_cols = [c for c in pred_df.columns if c.endswith("_correct")]

    relation_cols = ["rel1_correct", "rel2_correct"]
    attribute_cols = [c for c in correct_cols if c not in relation_cols]

    pred_df["exact_sentence_accuracy"] = pred_df["exact_match"].astype(float)
    pred_df["all_slots_joint_accuracy"] = pred_df[correct_cols].all(axis=1).astype(float)
    pred_df["mean_slot_accuracy"] = pred_df[correct_cols].mean(axis=1)
    pred_df["relation_accuracy"] = pred_df[relation_cols].mean(axis=1)
    pred_df["attribute_accuracy"] = pred_df[attribute_cols].mean(axis=1)

    pred_df["relation_error"] = ((pred_df["rel1_correct"] == 0) | (pred_df["rel2_correct"] == 0)).astype(int)
    pred_df["attribute_error"] = (pred_df[attribute_cols].sum(axis=1) < len(attribute_cols)).astype(int)

    return pred_df


# Calculate the average of each model’s accuracy across all categories on the test set
def summarise_overall(pred_df, model_name):
    summary = {
        "model": model_name,
        "n_samples": len(pred_df),
        "exact_sentence_accuracy": pred_df["exact_sentence_accuracy"].mean(),
        "all_slots_joint_accuracy": pred_df["all_slots_joint_accuracy"].mean(),
        "mean_slot_accuracy": pred_df["mean_slot_accuracy"].mean(),
        "relation_accuracy": pred_df["relation_accuracy"].mean(),
        "attribute_accuracy": pred_df["attribute_accuracy"].mean(),
    }
    return summary


# Group the statistics by a particular complexity variable
def summarise_by_group(pred_df, model_name, group_col):
    grouped = (
        pred_df.groupby(group_col)[
            [
                "exact_sentence_accuracy",
                "all_slots_joint_accuracy",
                "mean_slot_accuracy",
                "relation_accuracy",
                "attribute_accuracy",
            ]
        ]
        .mean()
        .reset_index()
    )

    counts = pred_df.groupby(group_col).size().reset_index(name="n_samples")
    grouped = grouped.merge(counts, on=group_col)
    grouped["model"] = model_name
    return grouped


# Save error examples from each model
def save_error_examples(pred_df, model_name, output_dir):
    show_cols = [
        "file_name",
        "ground_truth",
        "predicted_sentence",
        "object_count",
        "has_overlapping",
        "relation_pair_type",
        "rel1_correct",
        "rel2_correct",
        "relation_error",
        "attribute_error",
    ]

    relation_errors = pred_df[pred_df["relation_error"] == 1][show_cols].copy()
    attribute_errors = pred_df[pred_df["attribute_error"] == 1][show_cols].copy()

    relation_errors.head(30).to_csv(
        os.path.join(output_dir, f"{model_name}_relation_error_examples.csv"),
        index=False
    )
    attribute_errors.head(30).to_csv(
        os.path.join(output_dir, f"{model_name}_attribute_error_examples.csv"),
        index=False
    )


# analysis
def main():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("labels.csv not found.")

    labels_df = pd.read_csv(LABELS_PATH)
    complexity_df = build_complexity_table(labels_df)

    output_dir = "slot_structure_cnn_architecture_analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    overall_rows = []
    object_rows = []
    overlap_rows = []
    relation_rows = []

    available_models = {}

    for model_name, pred_path in MODEL_FILES.items():
        if os.path.exists(pred_path):
            available_models[model_name] = pred_path
        else:
            print(f"Skip {model_name}: file not found -> {pred_path}")

    if len(available_models) == 0:
        raise FileNotFoundError("No test_predictions.csv files were found.")

    for model_name, pred_path in available_models.items():
        print(f"Processing {model_name}...")

        pred_df = pd.read_csv(pred_path)
        pred_df = add_row_metrics(pred_df)

        merged = pred_df.merge(complexity_df, on=["file_name", "ground_truth"], how="left")

        overall_rows.append(summarise_overall(merged, model_name))

        object_rows.append(summarise_by_group(merged, model_name, "object_count"))
        overlap_rows.append(summarise_by_group(merged, model_name, "has_overlapping"))
        relation_rows.append(summarise_by_group(merged, model_name, "relation_pair_type"))

        save_error_examples(merged, model_name, output_dir)

    overall_df = pd.DataFrame(overall_rows)
    by_object_df = pd.concat(object_rows, ignore_index=True)
    by_overlap_df = pd.concat(overlap_rows, ignore_index=True)
    by_relation_df = pd.concat(relation_rows, ignore_index=True)

    overall_df.to_csv(os.path.join(output_dir, "overall_comparison.csv"), index=False)
    by_object_df.to_csv(os.path.join(output_dir, "by_object_count.csv"), index=False)
    by_overlap_df.to_csv(os.path.join(output_dir, "by_overlapping.csv"), index=False)
    by_relation_df.to_csv(os.path.join(output_dir, "by_relation_pair_type.csv"), index=False)

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()