# Archived Follow-up Experiments

This folder contains follow-up experiments tested after the main structured-output model.

These files are kept for transparency and documentation, but they are not the final recommended models for the report.

## Why these experiments were run

After building the structured slot-based model, several additional variants were tested to see whether relation prediction could be improved further.

This was necessary because relation prediction remained the main difficulty in the task, especially for directional pairs such as:

- above / below
- left / right

## Files in this folder

- `slot_structured_cnn_v3.py`
- `slot_structured_cnn_v21.py`
- `slot_structured_cnn_v22.py`
- `slot_structured_cnn_v23.py`

## What these variants tried to do

These follow-up models explored different ideas, such as:

- stronger or modified relation emphasis
- auxiliary relation factorisation
- more complex architectural changes
- alternative ways to guide relation learning

## Why they are archived instead of used as the final model

Although these variants were useful for analysis, they did not produce a more stable improvement than `slot_structured_cnn_v2.py` on the test set.

For that reason:

- `slot_structured_cnn_v2.py` remains the final recommended model
- the files in this folder should be treated as supplementary experiments or follow-up attempts

## How these files may still be useful

These archived experiments are still useful for:

- showing that multiple motivated improvements were explored
- supporting discussion of what did not work
- strengthening the error analysis and limitations section
- demonstrating that relation prediction is difficult and not solved simply by adding model complexity

## Recommendation for report use

These experiments should be used only as:

- supplementary evidence
- ablation/follow-up discussion
- examples of unsuccessful or inconclusive attempts

They should not replace the main result line based on the baseline, the initial structured model, and `slot_structured_cnn_v2.py`.
