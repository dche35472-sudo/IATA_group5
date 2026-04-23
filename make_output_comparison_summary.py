import json
import pandas as pd

# 1. Read updated baseline unified evaluation
with open("baseline_unified_metrics.json", "r", encoding="utf-8") as f:
    baseline_metrics = json.load(f)

# 2. Read controlled slot model metrics
with open("slot_model_outputs_controlled/metrics.json", "r", encoding="utf-8") as f:
    controlled_metrics = json.load(f)

# baseline metrics are stored under ["test"]
b = baseline_metrics["test"]

# controlled metrics are stored at the top level
c = controlled_metrics

rows = [
    {
        "model": "updated_baseline",
        "comparison_role": "main baseline for controlled output comparison",
        "exact_sentence_accuracy": b["exact_sentence_accuracy"],
        "all_slots_joint_accuracy": b["all_slots_joint_accuracy"],
        "mean_slot_accuracy": b["mean_slot_accuracy"],
        "relation_accuracy": b["relation_accuracy"],
        "attribute_accuracy": b["attribute_accuracy"],
    },
    {
        "model": "controlled_structured_slot",
        "comparison_role": "main model for controlled output comparison",
        "exact_sentence_accuracy": c["exact_sentence_accuracy"],
        "all_slots_joint_accuracy": c["all_slots_joint_accuracy"],
        "mean_slot_accuracy": c["mean_slot_accuracy"],
        "relation_accuracy": c["relation_accuracy"],
        "attribute_accuracy": c["attribute_accuracy"],
    },
]

df = pd.DataFrame(rows)

# Optional: round numeric columns for cleaner display
numeric_cols = [
    "exact_sentence_accuracy",
    "all_slots_joint_accuracy",
    "mean_slot_accuracy",
    "relation_accuracy",
    "attribute_accuracy",
]
df[numeric_cols] = df[numeric_cols].round(4)

# Save
output_path = "model_comparison_summary.csv"
df.to_csv(output_path, index=False)

print("Saved to:", output_path)
print(df)