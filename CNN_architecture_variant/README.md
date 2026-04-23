## 1. What this experimental axis studies

This part studies the following question:
Under the same slot-structured sentence representation (the one proposed in previous experiment), which CNN encoder architecture is the most effective?

All three models keep the following settings fixed:
* the same **11-slot structured output format**
* the same data split: **train / validation / test
* the same optimisation and training protocol: AdamW + scheduler + early stopping
* the same evaluation and output format: each model saves `metrics.json`, `training_history.csv`, `val/test_predictions.csv`, and relation confusion files.

Under these fixed conditions, this experimental axis compares 3 CNN architecture variants:
1. **SimpleCNN**: a shallow custom CNN encoder.
2. **DeeperCNN**: a deeper custom CNN encoder, same as the baseline encoder.
3. **ResNet18**: a ResNet18-based encoder.


## 2. Function of each code file

 `slot_simple_CNN.py`: trains and evaluates the **SimpleCNN** variant.
 `slot_deeper_CNN.py`: trains and evaluates the **DeeperCNN** variant.
 `slot_ResNet.py`: trains and evaluates the **ResNet18** variant.
 `slot_structure_cnn_architecture_analysis.py`: performs analysis on the test outputs of the three models.


## 3. Explanation of output files

### A. Model outputs (using `slot_simple_CNN_outputs` as example)

This folder is automatically generated after running `slot_simple_CNN.py`. The other two model output folders follow the same structure.

 `best_slot_simple_model.pt`: The saved best model checkpoint. It is selected during validation using the selection score.
 `metrics.json`: Stores the final metric summary
 `training_history.csv`: Stores epoch-level training history, including: training loss, validation metrics, selection score, learning rate.
 `val_predictions.csv`: Detailed prediction results for each validation sample.
 `test_predictions.csv`: Detailed prediction results for each test sample. This is the main input for later architecture comparison and error analysis.
 `val_rel1_confusion.csv` / `val_rel2_confusion.csv`: Confusion matrices for the two relation slots on the validation set.
 `test_rel1_confusion.csv` / `test_rel2_confusion.csv`: Confusion matrices for the two relation slots on the test set. These are used to inspect relation-type confusions.

### B. Analysis outputs: `slot_structure_cnn_architecture_analysis_outputs`

This folder is generated after running `slot_structure_cnn_architecture_analysis.py`. It contains the unified evaluation results.

 `overall_comparison.csv`: The overall performance comparison table for the three models. This is the main result table for architecture comparison.
 `by_object_count.csv`: Performance grouped by the number of objects in the image. Used to analyse how much performance drops as scene complexity increases.
 `by_overlapping.csv`: Performance grouped by whether the sample contains overlapping relations. Used to analyse whether overlapping makes the task harder.
 `by_relation_pair_type.csv`: Performance grouped by whether the two relations are the same or mixed. Used to analyse whether mixed relations are harder.
 `complexity_analysis_summary.csv`: A summary table for complexity analysis. 
 `error_type_summary.csv`: A summary table of error types for each model. Errors are divided into:`relation_only`,`attribute_only`,`mixed_error`,`both_correct`.
 `SimpleCNN_*_examples.csv` / `DeeperCNN_*_examples.csv` / `ResNet18_*_examples.csv`: Error example files for each model.