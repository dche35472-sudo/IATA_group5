import os
import json
import re
import pandas as pd


LABELS_PATH = "labels.csv"

MODEL_FILES = {
    "SimpleCNN": "slot_simple_CNN_outputs/test_predictions.csv",
    "DeeperCNN": "slot_deeper_CNN_outputs/test_predictions.csv",
    "ResNet18": "slot_ResNet_outputs/test_predictions.csv",
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


def build_complexity_table(labels_df):
    rows = []

    for _, row in labels_df.iterrows():
        file_name = row["file_name"]
        description = row["description"]
        objects = json.loads(row["objects"])

        parsed = parse_description(description)
        if parsed is None:
            continue

        # num of objects in this image
        object_count = len(objects)
        rel1 = parsed["rel1"]
        rel2 = parsed["rel2"]

        # if there ia an overlapping relation
        has_overlapping = int((rel1 == "overlapping") or (rel2 == "overlapping"))
        # relation type in this description
        relation_pair_type = "same_relation" if rel1 == rel2 else "mixed_relation"

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


# calculate performance metrics for every sample
def add_sample_metrics(pred_df):

    correct_cols = [c for c in pred_df.columns if c.endswith("_correct")]

    relation_cols = ["rel1_correct", "rel2_correct"]
    anchor_cols = ["anchor_size_correct", "anchor_color_correct", "anchor_shape_correct"]
    target1_cols = ["target1_size_correct", "target1_color_correct", "target1_shape_correct"]
    target2_cols = ["target2_size_correct", "target2_color_correct", "target2_shape_correct"]

    attribute_cols = anchor_cols + target1_cols + target2_cols

    # average accuracy across all 11 slots
    pred_df["slot_ave_accuracy"] = pred_df[correct_cols].mean(axis=1)

    # average relation accuracy
    pred_df["relation_slot_accuracy"] = pred_df[relation_cols].mean(axis=1)
    # correct only when both relations are correct
    pred_df["relation_pair_accuracy"] = pred_df[relation_cols].all(axis=1).astype(float)

    # average attribute accuracy
    pred_df["attribute_accuracy"] = pred_df[attribute_cols].mean(axis=1)

    # correct only when each object's descriptor (size+color+shape) completely correct
    pred_df["anchor_object_accuracy"] = pred_df[anchor_cols].all(axis=1).astype(float)
    pred_df["target1_object_accuracy"] = pred_df[target1_cols].all(axis=1).astype(float)
    pred_df["target2_object_accuracy"] = pred_df[target2_cols].all(axis=1).astype(float)
    pred_df["object_descriptor_accuracy"] = pred_df[
        ["anchor_object_accuracy", "target1_object_accuracy", "target2_object_accuracy"]
    ].mean(axis=1)

    # Error flags
    pred_df["relation_error"] = (pred_df["relation_pair_accuracy"] == 0).astype(int)
    pred_df["attribute_error"] = (pred_df[attribute_cols].sum(axis=1) < len(attribute_cols)).astype(int)

    pred_df["error_type"] = "both_correct"
    pred_df.loc[
        (pred_df["relation_error"] == 1) & (pred_df["attribute_error"] == 0),
        "error_type"
    ] = "relation_only"
    pred_df.loc[
        (pred_df["relation_error"] == 0) & (pred_df["attribute_error"] == 1),
        "error_type"
    ] = "attribute_only"
    pred_df.loc[
        (pred_df["relation_error"] == 1) & (pred_df["attribute_error"] == 1),
        "error_type"
    ] = "mixed_error"

    return pred_df


def summarise_overall(pred_df, model_name):
    summary = {
        "model": model_name,
        "n_samples": len(pred_df),
        "slot_ave_accuracy": pred_df["slot_ave_accuracy"].mean(),
        "object_descriptor_accuracy": pred_df["object_descriptor_accuracy"].mean(),
        "relation_pair_accuracy": pred_df["relation_pair_accuracy"].mean(),
        "relation_slot_accuracy": pred_df["relation_slot_accuracy"].mean(),
        "attribute_accuracy": pred_df["attribute_accuracy"].mean(),
    }
    return summary


def summarise_by_group(pred_df, model_name, group_col):
    grouped = (
        pred_df.groupby(group_col)[
            [
                "slot_ave_accuracy",
                "object_descriptor_accuracy",
                "relation_pair_accuracy",
                "relation_slot_accuracy",
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


def summarise_error_types(pred_df, model_name):
    counts = pred_df["error_type"].value_counts().reset_index()
    counts.columns = ["error_type", "n_samples"]
    counts["proportion"] = counts["n_samples"] / len(pred_df)
    counts["model"] = model_name
    return counts


def complexity_analysis_summary(by_object_df, by_overlap_df):
    rows = []

    for model in sorted(by_object_df["model"].unique()):
        obj_sub = by_object_df[by_object_df["model"] == model]
        overlap_sub = by_overlap_df[by_overlap_df["model"] == model]

        row = {"model": model}

        obj3 = obj_sub[obj_sub["object_count"] == 3]
        obj5 = obj_sub[obj_sub["object_count"] == 5]

        if len(obj3) == 1 and len(obj5) == 1:
            row["slot_ave_drop_3_to_5"] = float(
                obj3["slot_ave_accuracy"].iloc[0] - obj5["slot_ave_accuracy"].iloc[0]
            )
            row["object_descriptor_drop_3_to_5"] = float(
                obj3["object_descriptor_accuracy"].iloc[0] - obj5["object_descriptor_accuracy"].iloc[0]
            )
            row["relation_pair_drop_3_to_5"] = float(
                obj3["relation_pair_accuracy"].iloc[0] - obj5["relation_pair_accuracy"].iloc[0]
            )
        else:
            row["slot_ave_drop_3_to_5"] = None
            row["object_descriptor_drop_3_to_5"] = None
            row["relation_pair_drop_3_to_5"] = None

        no_overlap = overlap_sub[overlap_sub["has_overlapping"] == 0]
        yes_overlap = overlap_sub[overlap_sub["has_overlapping"] == 1]

        if len(no_overlap) == 1 and len(yes_overlap) == 1:
            row["overlap_penalty_slot_ave"] = float(
                no_overlap["slot_ave_accuracy"].iloc[0] - yes_overlap["slot_ave_accuracy"].iloc[0]
            )
            row["overlap_penalty_object_descriptor"] = float(
                no_overlap["object_descriptor_accuracy"].iloc[0] - yes_overlap["object_descriptor_accuracy"].iloc[0]
            )
            row["overlap_penalty_relation_pair"] = float(
                no_overlap["relation_pair_accuracy"].iloc[0] - yes_overlap["relation_pair_accuracy"].iloc[0]
            )
        else:
            row["overlap_penalty_slot_ave"] = None
            row["overlap_penalty_object_descriptor"] = None
            row["overlap_penalty_relation_pair"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def save_error_examples(pred_df, model_name, output_dir):
    show_cols = [
        "file_name",
        "ground_truth",
        "predicted_sentence",
        "object_count",
        "has_overlapping",
        "relation_pair_type",
        "anchor_object_accuracy",
        "target1_object_accuracy",
        "target2_object_accuracy",
        "rel1_correct",
        "rel2_correct",
        "relation_error",
        "attribute_error",
        "error_type",
    ]

    relation_errors = pred_df[pred_df["relation_error"] == 1][show_cols].copy()
    attribute_errors = pred_df[pred_df["attribute_error"] == 1][show_cols].copy()
    mixed_errors = pred_df[pred_df["error_type"] == "mixed_error"][show_cols].copy()

    relation_errors.head(30).to_csv(
        os.path.join(output_dir, f"{model_name}_relation_error_examples.csv"),
        index=False
    )
    attribute_errors.head(30).to_csv(
        os.path.join(output_dir, f"{model_name}_attribute_error_examples.csv"),
        index=False
    )
    mixed_errors.head(30).to_csv(
        os.path.join(output_dir, f"{model_name}_mixed_error_examples.csv"),
        index=False
    )


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
    error_rows = []

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
        pred_df = add_sample_metrics(pred_df)

        merged = pred_df.merge(complexity_df, on=["file_name", "ground_truth"], how="left")

        overall_rows.append(summarise_overall(merged, model_name))
        object_rows.append(summarise_by_group(merged, model_name, "object_count"))
        overlap_rows.append(summarise_by_group(merged, model_name, "has_overlapping"))
        relation_rows.append(summarise_by_group(merged, model_name, "relation_pair_type"))
        error_rows.append(summarise_error_types(merged, model_name))

        save_error_examples(merged, model_name, output_dir)

    overall_df = pd.DataFrame(overall_rows)
    by_object_df = pd.concat(object_rows, ignore_index=True)
    by_overlap_df = pd.concat(overlap_rows, ignore_index=True)
    by_relation_df = pd.concat(relation_rows, ignore_index=True)
    error_df = pd.concat(error_rows, ignore_index=True)
    complexity_analysis_df = complexity_analysis_summary(by_object_df, by_overlap_df)

    overall_df.to_csv(os.path.join(output_dir, "overall_comparison.csv"), index=False)
    by_object_df.to_csv(os.path.join(output_dir, "by_object_count.csv"), index=False)
    by_overlap_df.to_csv(os.path.join(output_dir, "by_overlapping.csv"), index=False)
    by_relation_df.to_csv(os.path.join(output_dir, "by_relation_pair_type.csv"), index=False)
    error_df.to_csv(os.path.join(output_dir, "error_type_summary.csv"), index=False)
    complexity_analysis_df.to_csv(os.path.join(output_dir, "complexity_analysis_summary.csv"), index=False)

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()