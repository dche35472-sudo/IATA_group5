# Text Representation / Output Format – Main Results

This folder contains the main result files for the text representation / output format comparison part of the project.

## Purpose of this module

The goal of this part is to compare different ways of representing the target sentence in the same image-to-sentence task.

The original baseline uses a multi-label bag-of-words style output, where the model predicts whether predefined words appear in the description.

The structured model instead uses a slot-based output format, where the sentence is decomposed into structured components such as:

- anchor size / colour / shape
- relation 1
- target 1 size / colour / shape
- relation 2
- target 2 size / colour / shape

This was designed to better match the structure of the dataset descriptions and to support more informative error analysis.

## Final recommended model

The final recommended model for this part is:

- `slot_structured_cnn_v2.py`

This version gave the most stable test performance among the structured-output variants and is the version recommended for the main comparison in the report.

## Files in this folder

- `metrics.json`  
  Main evaluation results for the final recommended structured model.

- `training_history.csv`  
  Training and validation history for the final model.

- `test_predictions.csv`  
  Test predictions from the final model.

- `test_rel1_confusion.csv`  
  Confusion results for the first relation slot.

- `test_rel2_confusion.csv`  
  Confusion results for the second relation slot.

- `val_predictions.csv`  
  Validation predictions for the final model.

- `val_rel1_confusion.csv`  
  Validation confusion results for the first relation slot.

- `val_rel2_confusion.csv`  
  Validation confusion results for the second relation slot.

- `best_slot_model_v2.pt`  
  Saved checkpoint of the final recommended model.

- `baseline_unified_metrics.json`  
  Unified evaluation results for the original baseline under the same evaluation framework.

- `baseline_unified_eval_details.csv`  
  Detailed baseline evaluation outputs.

- `model_comparison_summary.csv`  
  Summary table comparing baseline and structured-output variants.

## How these results should be used

For the main report, the recommended comparison is:

1. original baseline (`baseline_CNN.py`)
2. initial structured slot model
3. improved structured slot model (`slot_structured_cnn_v2.py`)

The main findings supported by these files are:

- the structured output format provides a better match to the sentence structure in the dataset
- a stronger visual encoder improves slot-level and attribute-level prediction
- relation prediction remains the main bottleneck, especially directional relations

## Notes

- Exact sentence match is a very strict metric in this task, because an error in any slot makes the full sentence incorrect.
- Therefore, slot-level accuracy, attribute accuracy, relation accuracy, and confusion analysis are also important for interpretation.
