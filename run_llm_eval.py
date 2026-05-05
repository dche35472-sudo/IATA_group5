"""
Commercial-LLM (vision-language model) comparison for the shape-description
task, using Google Gemini.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

HEAD_SPECS: Dict[str, List[str]] = {
    "anchor_size":    ["small", "medium", "big"],
    "anchor_color":   ["red", "blue", "green", "yellow", "black"],
    "anchor_shape":   ["circle", "square", "triangle"],
    "rel1":           ["left of", "right of", "above", "below", "overlapping"],
    "target1_size":   ["small", "medium", "big"],
    "target1_color":  ["red", "blue", "green", "yellow", "black"],
    "target1_shape":  ["circle", "square", "triangle"],
    "rel2":           ["left of", "right of", "above", "below", "overlapping"],
    "target2_size":   ["small", "medium", "big"],
    "target2_color":  ["red", "blue", "green", "yellow", "black"],
    "target2_shape":  ["circle", "square", "triangle"],
}
HEAD_ORDER = list(HEAD_SPECS.keys())

CLAUSE_RE = re.compile(
    r"a (?P<s>small|medium|big) "
    r"(?P<c>red|blue|green|yellow|black) "
    r"(?P<sh>circle|square|triangle) "
    r"is (?P<r>left of|right of|above|below|overlapping) "
    r"a (?P<ts>small|medium|big) "
    r"(?P<tc>red|blue|green|yellow|black) "
    r"(?P<tsh>circle|square|triangle)"
)


def parse_description(desc: str) -> Dict[str, str]:
    desc = str(desc).strip().lower()
    clauses = [c.strip() for c in desc.split("|")]

    parsed = []
    for clause in clauses:
        m = CLAUSE_RE.search(clause)
        if m:
            parsed.append({
                "anchor_size":  m.group("s"),
                "anchor_color": m.group("c"),
                "anchor_shape": m.group("sh"),
                "rel":          m.group("r"),
                "target_size":  m.group("ts"),
                "target_color": m.group("tc"),
                "target_shape": m.group("tsh"),
            })

    if len(parsed) < 2:
        raise ValueError(f"Need at least 2 clauses, got {len(parsed)}: {desc!r}")

    c1, c2 = None, None
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            p1, p2 = parsed[i], parsed[j]
            if (p1["anchor_size"]  == p2["anchor_size"] and
                p1["anchor_color"] == p2["anchor_color"] and
                p1["anchor_shape"] == p2["anchor_shape"]):
                c1, c2 = p1, p2
                break
        if c1:
            break

    if c1 is None:
        c1, c2 = parsed[0], parsed[1]

    return {
        "anchor_size":   c1["anchor_size"],
        "anchor_color":  c1["anchor_color"],
        "anchor_shape":  c1["anchor_shape"],
        "rel1":          c1["rel"],
        "target1_size":  c1["target_size"],
        "target1_color": c1["target_color"],
        "target1_shape": c1["target_shape"],
        "rel2":          c2["rel"],
        "target2_size":  c2["target_size"],
        "target2_color": c2["target_color"],
        "target2_shape": c2["target_shape"],
    }


SYSTEM_PROMPT = """You are a careful image describer for a research benchmark.

The image contains 3-5 simple coloured shapes on a noisy light background.
Vocabulary (use ONLY these words):
  size:     small | medium | big
  colour:   red | blue | green | yellow | black
  shape:    circle | square | triangle
  relation: left of | right of | above | below | overlapping

You must pick ONE anchor object and TWO different target objects, then output
EXACTLY this format on a single line, lower-case, no extra words:

a {size} {colour} {shape} is {relation} a {size} {colour} {shape} | a {size} {colour} {shape} is {relation} a {size} {colour} {shape}

Both clauses share the SAME anchor object (size/colour/shape repeated).
Pick relations whose direction is visually unambiguous.

Return ONLY that one line, nothing else."""

USER_PROMPT = "Describe this image using the format above."


def _make_config(model: str):
    from google.genai import types

    base = dict(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=200,
        temperature=0.0,
    )

    m = model.lower()
    if "2.5" in m:
        base["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    elif "3.1-flash-lite" in m:
        base["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")

    return types.GenerateContentConfig(**base)


def call_gemini(image_path: Path, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    with open(image_path, "rb") as fh:
        image_bytes = fh.read()

    config = _make_config(model)

    # Unlimited 503 retries with capped backoff — waits until demand clears
    attempt = 0
    while True:
        try:
            rsp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    USER_PROMPT,
                ],
                config=config,
            )
            text = (rsp.text or "").strip()

            if not text:
                wait = min(15 * (2 ** attempt), 120)
                print(f"  [warn] empty response attempt {attempt+1}, waiting {wait}s...")
                time.sleep(wait)
                attempt += 1
                continue

            return text

        except Exception as e:
            err = str(e)

            if "503" in err or "UNAVAILABLE" in err or "high demand" in err.lower():
                # Retry indefinitely with capped backoff until server recovers
                wait = min(30 * (2 ** attempt), 300)  # caps at 5 minutes
                print(f"  [warn] 503 high demand attempt {attempt+1}, waiting {wait}s...")
                time.sleep(wait)
                attempt += 1
                continue

            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                wait = min(15 * (2 ** attempt), 300)
                print(f"  [warn] rate limit attempt {attempt+1}, waiting {wait}s...")
                time.sleep(wait)
                attempt += 1
                if attempt >= 8:
                    return f"<<error: quota exhausted after {attempt} retries>>"
                continue

            # Any other error — fail immediately
            return f"<<error: {e}>>"


def score_row(gold: Dict[str, str], pred: Optional[Dict[str, str]]) -> Dict[str, float]:
    if pred is None:
        per_slot = {f"match_{k}": 0 for k in HEAD_ORDER}
    else:
        per_slot = {f"match_{k}": int(pred[k] == gold[k]) for k in HEAD_ORDER}

        # Order-invariant relation scoring: correct if both relations match in any order
        gold_rels = {gold["rel1"], gold["rel2"]}
        pred_rels = {pred["rel1"], pred["rel2"]}
        if gold_rels == pred_rels:
            per_slot["match_rel1"] = 1
            per_slot["match_rel2"] = 1

    matches = list(per_slot.values())
    rel_matches = [per_slot["match_rel1"], per_slot["match_rel2"]]
    attr_matches = [v for k, v in per_slot.items() if not k.startswith("match_rel")]

    per_slot["mean_slot_accuracy"]    = sum(matches) / len(matches)
    per_slot["relation_accuracy"]     = sum(rel_matches) / 2
    per_slot["attribute_accuracy"]    = sum(attr_matches) / 9
    per_slot["all_slots_joint_match"] = int(all(matches))
    return per_slot


def parse_loose(raw: str) -> Optional[Dict[str, str]]:
    for line in raw.splitlines():
        line = line.strip().lower().rstrip(".").strip()
        line = re.sub(r"^[-*>`'\"]+\s*", "", line)
        line = re.sub(r"\s*[`'\"]+$", "", line)
        try:
            return parse_description(line)
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv",  required=True)
    ap.add_argument("--image_dir",   required=True)
    ap.add_argument("--n_samples",   type=int, default=100)
    ap.add_argument("--sample_seed", type=int, default=123)
    ap.add_argument("--model",       default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--sleep",       type=float, default=5.0,
                    help="Seconds between API calls (free tier: keep <=12 RPM).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    _, test_df = train_test_split(df, test_size=0.15, random_state=42)
    sample = test_df.sample(
        n=args.n_samples, random_state=args.sample_seed
    ).reset_index(drop=True)
    sample[["image_id", "file_name", "description"]].to_csv(
        out_dir / "sample_index.csv", index=False
    )

    image_dir    = Path(args.image_dir)
    raw_path     = out_dir / "raw_responses.jsonl"
    partial_path = out_dir / "predictions_partial.csv"

    done: set = set()
    pred_rows: List[Dict] = []
    if partial_path.exists():
        prev = pd.read_csv(partial_path)
        prev = prev[prev["pred_sentence"] != "<<unparseable>>"].copy()
        pred_rows = prev.to_dict(orient="records")
        done = set(prev["file_name"].tolist())
        print(f"[resume] kept {len(done)} previously-good rows")

    raw_fh = open(raw_path, "a")

    for idx, row in sample.iterrows():
        if row["file_name"] in done:
            continue

        image_path = image_dir / row["file_name"]
        gold_sent  = str(row["description"]).strip().lower()

        try:
            gold_slots = parse_description(gold_sent)
        except ValueError as e:
            print(f"[{idx+1:>3d}/{len(sample)}] SKIP {row['file_name']} — bad gold: {e}")
            continue

        raw = call_gemini(image_path, args.model)

        pred_slots = parse_loose(raw)
        pred_sent = (
            "a {anchor_size} {anchor_color} {anchor_shape} is {rel1} "
            "a {target1_size} {target1_color} {target1_shape} | "
            "a {anchor_size} {anchor_color} {anchor_shape} is {rel2} "
            "a {target2_size} {target2_color} {target2_shape}"
        ).format(**pred_slots) if pred_slots is not None else "<<unparseable>>"

        scored = score_row(gold_slots, pred_slots)

        out_row = {
            "file_name":     row["file_name"],
            "gold_sentence": gold_sent,
            "pred_sentence": pred_sent,
            "raw_response":  raw,
        }
        for k in HEAD_ORDER:
            out_row[f"gold_{k}"] = gold_slots[k]
            out_row[f"pred_{k}"] = pred_slots[k] if pred_slots else ""
        out_row.update(scored)
        pred_rows.append(out_row)

        raw_fh.write(json.dumps({
            "file_name": row["file_name"],
            "gold":      gold_sent,
            "raw":       raw,
        }) + "\n")
        raw_fh.flush()

        pd.DataFrame(pred_rows).to_csv(partial_path, index=False)

        print(f"[{idx+1:>3d}/{len(sample)}] {row['file_name']}  "
              f"slot_acc={scored['mean_slot_accuracy']:.2f}  "
              f"rel_acc={scored['relation_accuracy']:.2f}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    raw_fh.close()

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "n_samples":                len(pred_df),
        "exact_sentence_accuracy":  float((pred_df["gold_sentence"] == pred_df["pred_sentence"]).mean()),
        "all_slots_joint_accuracy": float(pred_df["all_slots_joint_match"].mean()),
        "mean_slot_accuracy":       float(pred_df["mean_slot_accuracy"].mean()),
        "relation_accuracy":        float(pred_df["relation_accuracy"].mean()),
        "attribute_accuracy":       float(pred_df["attribute_accuracy"].mean()),
        "parse_failure_rate":       float((pred_df["pred_sentence"] == "<<unparseable>>").mean()),
        "per_slot_accuracy": {
            k: float(pred_df[f"match_{k}"].mean()) for k in HEAD_ORDER
        },
        "backend": "gemini",
        "model":   args.model,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== AGGREGATE METRICS ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()