# Text Representation / Output Format – Main Results

This folder contains the main result files for the **text representation / output format comparison** part of the project.

## Purpose of this module

The aim of this part is to compare different ways of representing the target sentence in the same image-to-sentence task.

The updated baseline uses a **multi-label bag-of-words style output**, where the model predicts whether predefined words appear in the description.

The structured model instead uses a **slot-based output format**, where the sentence is decomposed into structured components such as:

- anchor size / colour / shape
- relation 1
- target 1 size / colour / shape
- relation 2
- target 2 size / colour / shape

This was designed to better match the structure of the dataset descriptions and to support more informative error analysis.

## Main comparison in this folder

The main controlled comparison in this folder is:

1. **updated baseline** (`baseline_CNN.py`)
2. **controlled structured slot model**

In this comparison, the encoder and preprocessing settings are aligned as closely as possible, so the main difference is the **output representation** rather than the visual backbone.

## Final recommended model for this module

The final recommended model for this specific module is the **controlled structured slot model**, which is used for the controlled output representation comparison.

## Files in this folder

- `metrics.json`  
  Main evaluation results for the controlled structured slot model.

- `training_history.csv`  
  Training history for the controlled structured slot model.

- `test_predictions.csv`  
  Test predictions from the controlled structured slot model.

- `test_rel1_confusion.csv`  
  Confusion results for the first relation slot.

- `test_rel2_confusion.csv`  
  Confusion results for the second relation slot.

- `best_slot_model_controlled.pt`  
  Saved checkpoint of the controlled structured slot model.

- `baseline_unified_metrics.json`  
  Unified evaluation results for the updated baseline under the same evaluation framework.

- `baseline_unified_eval_details.csv`  
  Detailed unified evaluation outputs for the updated baseline.

- `model_comparison_summary.csv`  
  Summary table comparing the updated baseline and the controlled structured slot model.

## Main findings supported by these files

The main findings from this controlled comparison are:

- the structured slot-based output matches the dataset sentence structure better than the bag-of-words baseline output
- under the same encoder setting, the structured output performs better on slot-level, attribute-level, and relation-level metrics
- relation prediction remains the main bottleneck, especially directional relations

## Notes

- Exact sentence match is a very strict metric in this task, because an error in any slot makes the full sentence incorrect.
- Therefore, slot-level accuracy, attribute accuracy, relation accuracy, and confusion analysis are also important for interpretation.
- The baseline unified evaluation uses an intentionally optimistic mapping from bag-of-words predictions to the structured two-clause format, in order to make the comparison fairer to the baseline.
