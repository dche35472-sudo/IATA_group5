import json
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

GT_RE = re.compile(
    r"^a (\w+) (\w+) (\w+) is (left of|right of|above|below|overlapping) a (\w+) (\w+) (\w+) \| "
    r"a (\w+) (\w+) (\w+) is (left of|right of|above|below|overlapping) a (\w+) (\w+) (\w+)$"
)
CLAUSE_RE = re.compile(
    r"^a (\w+) (\w+) (\w+) is (left of|right of|above|below) a (\w+) (\w+) (\w+)$"
)
SLOT_KEYS = [
    "anchor_size", "anchor_color", "anchor_shape",
    "rel1", "target1_size", "target1_color", "target1_shape",
    "rel2", "target2_size", "target2_color", "target2_shape",
]
ATTRIBUTE_KEYS = [k for k in SLOT_KEYS if not k.startswith("rel")]


def parse_ground_truth(text: str):
    text = str(text).strip().lower()
    m = GT_RE.match(text)
    if not m:
        return None
    g = m.groups()
    return {
        "anchor_size": g[0], "anchor_color": g[1], "anchor_shape": g[2],
        "rel1": g[3], "target1_size": g[4], "target1_color": g[5], "target1_shape": g[6],
        "rel2": g[10], "target2_size": g[11], "target2_color": g[12], "target2_shape": g[13],
    }


def parse_predicted_clauses(text: str):
    text = str(text).strip().lower()
    if not text or text == "no objects detected":
        return []
    clauses = [c.strip() for c in text.split("|")]
    parsed = []
    for clause in clauses:
        m = CLAUSE_RE.match(clause)
        if not m:
            continue
        g = m.groups()
        parsed.append(
            {
                "subject": (g[0], g[1], g[2]),
                "relation": g[3],
                "object": (g[4], g[5], g[6]),
                "text": clause,
            }
        )
    return parsed


def slots_to_sentence(slots: dict):
    if slots is None:
        return None
    if any(slots.get(k) is None for k in SLOT_KEYS):
        return None
    return (
        f"a {slots['anchor_size']} {slots['anchor_color']} {slots['anchor_shape']} is {slots['rel1']} "
        f"a {slots['target1_size']} {slots['target1_color']} {slots['target1_shape']} | "
        f"a {slots['anchor_size']} {slots['anchor_color']} {slots['anchor_shape']} is {slots['rel2']} "
        f"a {slots['target2_size']} {slots['target2_color']} {slots['target2_shape']}"
    )


def make_candidates_from_baseline_output(text: str):
    clauses = parse_predicted_clauses(text)
    if not clauses:
        return []

    grouped = {}
    for cl in clauses:
        grouped.setdefault(cl["subject"], []).append(cl)

    candidates = []
    for subject, subject_clauses in grouped.items():
        if len(subject_clauses) >= 2:
            for c1, c2 in itertools.permutations(subject_clauses, 2):
                candidates.append(
                    {
                        "anchor_size": subject[0], "anchor_color": subject[1], "anchor_shape": subject[2],
                        "rel1": c1["relation"],
                        "target1_size": c1["object"][0], "target1_color": c1["object"][1], "target1_shape": c1["object"][2],
                        "rel2": c2["relation"],
                        "target2_size": c2["object"][0], "target2_color": c2["object"][1], "target2_shape": c2["object"][2],
                    }
                )
        else:
            c1 = subject_clauses[0]
            candidates.append(
                {
                    "anchor_size": subject[0], "anchor_color": subject[1], "anchor_shape": subject[2],
                    "rel1": c1["relation"],
                    "target1_size": c1["object"][0], "target1_color": c1["object"][1], "target1_shape": c1["object"][2],
                    "rel2": None, "target2_size": None, "target2_color": None, "target2_shape": None,
                }
            )
    return candidates


def evaluate_one(pred_slots: dict, gold_slots: dict):
    if pred_slots is None or gold_slots is None:
        return {
            "exact_sentence_accuracy": 0.0,
            "all_slots_joint_accuracy": 0.0,
            "mean_slot_accuracy": 0.0,
            "relation_accuracy": 0.0,
            "attribute_accuracy": 0.0,
        }

    exact = float(slots_to_sentence(pred_slots) == slots_to_sentence(gold_slots))
    joint = float(all(pred_slots.get(k) == gold_slots[k] for k in SLOT_KEYS))
    mean_slot = float(np.mean([pred_slots.get(k) == gold_slots[k] for k in SLOT_KEYS]))
    rel_acc = float(np.mean([pred_slots.get("rel1") == gold_slots["rel1"], pred_slots.get("rel2") == gold_slots["rel2"]]))
    attr_acc = float(np.mean([pred_slots.get(k) == gold_slots[k] for k in ATTRIBUTE_KEYS]))
    return {
        "exact_sentence_accuracy": exact,
        "all_slots_joint_accuracy": joint,
        "mean_slot_accuracy": mean_slot,
        "relation_accuracy": rel_acc,
        "attribute_accuracy": attr_acc,
    }


def main():
    results_path = Path("baseline/baseline_overlapping_results.csv")
    if not results_path.exists():
        raise FileNotFoundError("baseline_overlapping_results.csv not found in current directory")

    df = pd.read_csv(results_path)
    per_row = []
    parseable_rows = 0

    for _, row in df.iterrows():
        gold = parse_ground_truth(row["ground_truth"])
        candidates = make_candidates_from_baseline_output(row["baseline_output"])
        if candidates:
            parseable_rows += 1
            scored = [evaluate_one(c, gold) for c in candidates]
            best_idx = max(
                range(len(scored)),
                key=lambda i: (
                    scored[i]["exact_sentence_accuracy"],
                    scored[i]["all_slots_joint_accuracy"],
                    scored[i]["mean_slot_accuracy"],
                    scored[i]["relation_accuracy"],
                    scored[i]["attribute_accuracy"],
                ),
            )
            best_metrics = scored[best_idx]
            best_candidate = candidates[best_idx]
        else:
            best_metrics = evaluate_one(None, gold)
            best_candidate = None

        per_row.append(
            {
                "file_name": row["file_name"],
                "ground_truth": row["ground_truth"],
                "baseline_output": row["baseline_output"],
                "decoded_anchor": None if best_candidate is None else f"{best_candidate['anchor_size']} {best_candidate['anchor_color']} {best_candidate['anchor_shape']}",
                **best_metrics,
            }
        )

    detail_df = pd.DataFrame(per_row)
    detail_df.to_csv("baseline_unified_eval_details.csv", index=False)

    metrics = {
        "test": {
            "exact_sentence_accuracy": float(detail_df["exact_sentence_accuracy"].mean()),
            "all_slots_joint_accuracy": float(detail_df["all_slots_joint_accuracy"].mean()),
            "mean_slot_accuracy": float(detail_df["mean_slot_accuracy"].mean()),
            "relation_accuracy": float(detail_df["relation_accuracy"].mean()),
            "attribute_accuracy": float(detail_df["attribute_accuracy"].mean()),
        },
        "notes": {
            "evaluation_style": "optimistic best-candidate mapping from bag-of-words output to the structured two-clause slot format",
            "parseable_prediction_rows": int(parseable_rows),
            "total_rows": int(len(detail_df)),
            "prediction_parse_rate": float(parseable_rows / len(detail_df)),
            "important_limitations": [
                "baseline vocabulary does not include overlapping",
                "baseline decoder generates reciprocal sentences rather than the anchor-repeated gold template",
                "this mapping is intentionally optimistic to make comparison fairer to the baseline",
            ],
        },
    }

    with open("baseline_unified_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Saved baseline_unified_metrics.json and baseline_unified_eval_details.csv")


if __name__ == "__main__":
    main()
