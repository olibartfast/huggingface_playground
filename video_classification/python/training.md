# Model Training Guide

This document provides complete, runnable Python scripts for training custom classifiers using models exported from this project.

## Strategy 1: Linear Probe (The "Embeddings" Approach)

**Best for**: Fast training, small datasets, or when you don't have GPU resources to finetune a transformer.
**Concept**: We freeze the heavy video model (Base) and just train a tiny classifier on its output vectors.

**Prerequisites**:
1.  Run `export.py` to get `videomae_embeddings.onnx` (or similar).
2.  Pass your dataset through this ONNX model to save embeddings to disk (e.g. `X_train.npy`, `y_train.npy`).

**Training Script (`train_linear_probe.py`)**:

```python
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_linear_probe():
    print("🚀 Starting Linear Probe Training...")
    
    # 1. Load your embeddings
    # Assume X is (N_samples, Hidden_Dim) e.g. (1000, 768)
    # If your model outputs (N, Patches, Dim), average across patches first!
    print("📦 Loading data...")
    try:
        X = np.load("dataset_embeddings.npy")
        y = np.load("dataset_labels.npy")
    except FileNotFoundError:
        print("⚠️  No data found. Generating DUMMY data for specific demo.")
        X = np.random.randn(1000, 768) # 1000 videos, 768 dim vectors
        y = np.random.randint(0, 5, 1000) # 5 classes

    # 2. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")

    # 3. Train Classifier
    # 'sag' solver is faster for large datasets
    clf = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=1000, verbose=1)
    clf.fit(X_train, y_train)

    # 4. Evaluate
    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"✅ Validation Accuracy: {acc:.4f}")

    # 5. Save
    joblib.dump(clf, "custom_classifier.pkl")
    print("💾 Model saved to custom_classifier.pkl")

if __name__ == "__main__":
    train_linear_probe()
```

## Strategy 2: Full Finetuning (The "PyTorch" Approach)

**Best for**: Maximum accuracy, specialized domains (e.g. medical video), provided you have a GPU.
**Concept**: We take a pretrained model, surgically replace its "head" (the output layer), and retrain it.

**Training Script (`finetune_model.py`)**:

```python
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification, TrainingArguments, Trainer
from datasets import load_dataset

def finetune_video_model():
    # CONFIGURATION
    MODEL_ID = "MCG-NJU/videomae-base" # Starting from a Base model
    OUTPUT_DIR = "./my_finetuned_model"
    NUM_CLASSES = 5
    LABELS = ["Walk", "Run", "Swim", "Jump", "Sit"]
    
    # 1. Prepare Model with NEW HEAD
    # ignore_mismatched_sizes=True allows us to load a base model and add a random head
    model = AutoModelForVideoClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        id2label={i: c for i, c in enumerate(LABELS)},
        label2id={c: i for i, c in enumerate(LABELS)}
    )
    
    # 2. Load Processor
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    # 3. Dummy Dataset (Replace with your own 'imagefolder' dataset)
    # Dataset structure: generator -> (pixel_values, labels)
    print("⚠️  Using dummy data generator. Replace with 'load_dataset(\"imagefolder\", data_dir=...)'")
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx):
            # Random video: 16 frames, 3 channels, 224x224
            return {
                "pixel_values": torch.randn(16, 3, 224, 224),
                "labels": torch.tensor(0) # All class 0
            }
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()

    # 4. Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2, # Low batch size for video (VRAM heavy)
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor, # Just typically passed for interface consistency
    )

    # 6. Train & Save
    print("🚀 Starting Finetuning...")
    trainer.train()
    
    print(f"💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Done! You can now run 'export.py' on this new folder.")

if __name__ == "__main__":
    finetune_video_model()
```
