# README.md (Task 5: Spatial Relationship Detection)

## 1. What this experimental axis studies

This part studies the following question:
How effectively can a custom CNN architecture capture complex visual relationships (specifically **overlapping**, **spatial directions**, and **object attributes**) from synthetic geometric images?

The experiment is centered on a **BaselineCNN** architecture designed to encode images and decode them into a multi-label set of vocabulary tags.

The model settings are as follows:
* **Output Format**: Multi-label classification across a **16-word vocabulary** (shapes, colors, sizes, and relations).
* **Data Scale**: Full dataset training (approx. 15,000 samples).
* **Architecture**: A 6-layer structure (4 Convolutional layers for feature extraction + 2 Fully Connected layers for classification).
* **Evaluation Metric**: **Exact Match Accuracy (Subset Accuracy)** — the model must predict all tags correctly for a sample to be considered "correct".

## 2. Function of each code file

* `baseline_CNN.py`: The core script that defines the model, data loading pipeline, and training/evaluation loop.
    * **Encoder**: 4 Conv layers that hierarchically extract features from raw pixels to complex spatial relationships.
    * **Classifier**: 2 FC layers that map consolidated features to vocabulary probabilities.
    * **Sentence Generator**: A logic-based post-processor that converts predicted tags into human-readable reciprocal sentences (e.g., "A is above B | B is below A").

## 3. Explanation of output files

### A. Model outputs & Logs

* `baseline_overlapping_results.csv`: The primary output file generated after evaluation. It contains:
    * `file_name`: The identifier of the test image.
    * `ground_truth`: The original description provided in the dataset.
    * `baseline_output`: The reciprocal sentences generated based on the model's predictions.
* **Console Logs**: During execution, the script outputs:
    * **Training Loss**: Per-epoch loss to monitor convergence.
    * **Final Test Accuracy**: The "Exact Match" percentage, representing the model's ability to perfectly understand the scene.

## 4. Model Architecture Detail (4 Conv + 2 FC)

The `BaselineCNN` utilizes a deep hierarchical structure to solve the **"Semantic Gap"**:

1.  **Feature Extraction (Encoder)**:
    * **Layers 1-2**: Detect low-level features such as color patches and basic edges (lines, curves).
    * **Layers 3-4**: Capture high-level semantics. As the **receptive field** increases through max-pooling, these layers recognize complete shapes and complex spatial intersections (critical for the `overlapping` tag).
2.  **Classification (Decoder)**:
    * **Flatten Layer**: Converts the $256 \times 4 \times 4$ feature map into a 4096-dimensional vector.
    * **FC Layer 1 (Hidden)**: A 512-node layer with **ReLU** and **Dropout (0.4)**. It performs "logical fusion" of the 4096 visual signals.
    * **FC Layer 2 (Output)**: A 16-node layer with **Sigmoid** activation, outputting the probability for each word in the `VOCAB`.
