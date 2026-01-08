# Model Export Instructions

This document explains how to export Hugging Face video models to ONNX format using the provided `export.py` script. The script handles both classification models and feature extractors (embeddings).

## Utilization

The `export.py` script simplifies the export process by automatically detecting the model type and standardizing the output:

1.  **Embeddings / Feature Extraction**:
    *   **Detection**: Models without a classification head (typically base models).
    *   **Output**: The script extracts the `last_hidden_state`.
    *   **Use Case**: Suitable for downstream tasks like video similarity search, clustering, or feeding into a separate classifier.
    *   **Output Shape**: `[Batch, Patches, Vector_Dim]` (e.g., `[1, 1568, 768]`).
    *   **File Suffix**: `_embed.onnx`

2.  **Video Classification**:
    *   **Detection**: Models with a `ForVideoClassification` architecture.
    *   **Output**: The script extracts the `logits`.
    *   **Use Case**: Direct classification into predefined categories (e.g., Kinetics-400).
    *   **Output Shape**: `[Batch, Num_Classes]` (e.g., `[1, 400]`).
    *   **File Suffix**: `_class.onnx`

## Model Terminology: Base vs Finetuned

Understanding the difference between model types is crucial for selecting the right export mode:

*   **Base Models** (e.g., `videomae-base`, `vjepa-base`):
    *   **Description**: These models are pre-trained on large datasets (like K400 or SSv2) using self-supervised learning *without* labels. They learn to understand video dynamics but don't know specific categories.
    *   **Components**: They consist only of the transformer encoder (backbone). They typically do **not** have a classification head.
    *   **Export Behavior**: Since they lack a classification layer, `export.py` defaults to exporting their **embeddings** (`last_hidden_state`). You use these embeddings to train your own lightweight classifiers.

*   **Finetuned Models** (e.g., `videomae-base-finetuned-kinetics`):
    *   **Description**: These are base models that have been further trained (finetuned) on a labeled dataset (like Kinetics-400) to recognize specific actions.
    *   **Components**: They include the base backbone *plus* a classification head (linear layer) on top.
    *   **Export Behavior**: The script detects the classification architecture and exports the **logits** (raw scores for each class). These are ready for immediate inference.

### FAQ: Which class does a "Base" model predict?

**Short Answer: None.**

A base model does not know what a "cat" or "playing tennis" is. It only outputs a mathematical representation (embedding) of the video.

To get a class from a base model, you must:
1.  **Export Embeddings**: Use this script to get the vectors.
2.  **Train a Classifier**: Train a simple Logistic Regression or Linear SVM on top of these vectors using your own labeled data.
3.  **Predict**: During inference, run `Video -> Base Model -> Embedding -> Your Classifier -> Class`.

This determines "which class belongs" based on *your* specific training data. If you want out-of-the-box classification, you **must** use a model with `finetuned` in the name.

## How to Run

1.  **Configure Models**:
    Open `export.py` and modify the `models_to_export` list in the `main` block to include the Hugging Face model IDs you wish to export.

    ```python
    models_to_export = [
        "MCG-NJU/videomae-base",           # Exports as embeddings
        "MCG-NJU/videomae-base-finetuned-kinetics", # Exports as classifier
        # Add your models here
    ]
    ```

2.  **Execute Script**:
    Run the script from the command line:

    ```bash
    python export.py
    ```

    ```bash
    python export.py
    ```

## Training Custom Classifiers

For instructions on how to train your own classifiers (using Linear Probes or Full Finetuning) using these models, please refer to:
👉 **[python/training.md](training.md)**

## Technical Details

The script uses a `UniversalVideoWrapper` class to solve common export issues:


*   **Tuple Unpacking**: Automatically handles models that return tuples or specific output objects (like `ImageClassifierOutput`) by extracting the relevant tensor.
*   **Dynamic Axes**: Sets batch size as a dynamic axis, allowing flexible batch sizes during inference.
*   **Opset Version**: Uses opset 14 for broad compatibility with Transformer models.