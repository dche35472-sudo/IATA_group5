# Text Representation / Output Format – Main Results

This folder contains the main result files for the **text representation / output format comparison** part of the project.

## Purpose of this module

The aim of this module is to compare different ways of representing the target sentence in the same image-to-sentence task.

The updated baseline uses a **multi-label bag-of-words output**, where the model predicts whether predefined words appear in the description. This provides a simple and useful baseline, but it does not explicitly represent the structure of the target sentence.

The structured model instead uses a **slot-based output format**, where each sentence is decomposed into structured components:

- anchor size / colour / shape
- relation 1
- target 1 size / colour / shape
- relation 2
- target 2 size / colour / shape

This representation was designed to better match the dataset descriptions and to support more detailed error analysis.

## Main controlled comparison

The main comparison in this module is between:

1. **updated baseline** (`baseline_CNN.py`)
2. **controlled structured slot model** (`slot_structured_cnn_controlled.py`)

The controlled structured slot model uses the same general encoder style, image size, and train/test split setting as the updated baseline. This means that the main experimental difference is the **output representation**, rather than the visual backbone.

## Final recommended model for this module

The final recommended model for this module is the **controlled structured slot model**.

This does not mean that the full sentence prediction task is solved. Exact sentence accuracy remains very low because a sentence is only counted as correct when every slot is correct. However, the structured slot model gives better partial prediction performance and provides more interpretable error analysis than the bag-of-words baseline.

## Main results

The corrected comparison results are:

| Model | Exact sentence accuracy | All-slots joint accuracy | Mean slot accuracy | Relation accuracy | Attribute accuracy |
|---|---:|---:|---:|---:|---:|
| updated baseline | 0.0000 | 0.0000 | 0.2533 | 0.2100 | 0.2629 |
| controlled structured slot | 0.0004 | 0.0004 | 0.4438 | 0.3107 | 0.4734 |

After correcting the baseline evaluation to parse `overlapping` predictions, the baseline scores increased, but the controlled structured slot model still performs better on mean slot accuracy, relation accuracy, and attribute accuracy.

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
  Saved checkpoint of the controlled structured slot model. The filename is kept for consistency with earlier outputs, but this checkpoint represents the final training epoch rather than a validation-selected best epoch.

## Related files stored in the project root

The following files are related to this module but are stored in the project root rather than in this folder:

- `baseline_CNN.py`  
  Updated bag-of-words baseline model used for the controlled comparison.

- `slot_structured_cnn_controlled.py`  
  Controlled structured slot model script. The current version includes seed initialisation for better reproducibility.

- `baseline_unified_eval.py`  
  Script used to evaluate the baseline under the same structured evaluation framework. The corrected version parses `overlapping` predictions.

- `baseline_unified_metrics.json`  
  Unified evaluation results for the updated baseline.

- `baseline_unified_eval_details.csv`  
  Detailed unified evaluation outputs for the updated baseline.

- `model_comparison_summary.csv`  
  Summary table comparing the updated baseline and the controlled structured slot model.

- `make_output_comparison_summary.py`  
  Script used to generate the comparison summary table.

- `rel1_confusion.png`  
  Visual confusion matrix for the first relation slot.

- `rel2_confusion.png`  
  Visual confusion matrix for the second relation slot.

## Main findings supported by these files

The main findings from this controlled comparison are:

- the structured slot-based output matches the dataset sentence structure better than the bag-of-words baseline output
- after correcting the baseline evaluation, the structured slot model still achieves higher mean slot, relation, and attribute accuracy
- exact sentence accuracy remains very low for both models, showing that full sentence prediction is still difficult
- relation prediction remains the main bottleneck, especially directional relations such as `above` / `below` and `left of` / `right of`
- `overlapping` is comparatively easier to detect than some directional relations in the structured model outputs

## Notes

- Exact sentence match is a very strict metric in this task, because an error in any single slot makes the full sentence incorrect.
- Therefore, slot-level accuracy, attribute accuracy, relation accuracy, and confusion analysis are important for interpreting model behaviour.
- The baseline unified evaluation uses an intentionally optimistic mapping from bag-of-words predictions to the structured two-clause format, in order to make the comparison fairer to the baseline.
- The controlled structured slot model is now seeded for reproducibility. However, because no validation split is used, the saved checkpoint should be interpreted as the final epoch checkpoint rather than a validation-selected best model.
