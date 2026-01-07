import torch
import argparse
from transformers import AutoConfig, VideoMAEModel, VideoMAEForVideoClassification

# --- 1. HELPER CLASS FOR EMBEDDINGS ---
# Wraps the base model to ensure ONNX returns a clean tensor, not a dictionary
class FeatureExtractorWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x).last_hidden_state

def export_videomae(model_id, output_path=None):
    print(f"--- Inspecting {model_id} ---")
    
    # --- 2. SMART DETECTION LOGIC ---
    try:
        config = AutoConfig.from_pretrained(model_id)
        archs = config.architectures if config.architectures else []
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # default to embeddings if unknown
    mode = "embeddings" 
    
    if "VideoMAEForVideoClassification" in archs:
        mode = "classification"
        print(f"🔍 Detection: Found '{archs[0]}'. This is a FINE-TUNED model.")
    elif "VideoMAEModel" in archs:
        mode = "embeddings"
        print(f"🔍 Detection: Found '{archs[0]}'. This is a BASE model.")
    else:
        print(f"⚠️ Warning: Architecture '{archs}' not explicitly recognized. Defaulting to Embeddings.")

    # --- 3. LOAD THE CORRECT MODEL ---
    if mode == "classification":
        model = VideoMAEForVideoClassification.from_pretrained(model_id)
        output_name = "logits"
        filename = output_path or "videomae_classification.onnx"
        print(f"📊 Task: Exporting Classification Head ({model.config.num_labels} classes)")
    else:
        base_model = VideoMAEModel.from_pretrained(model_id)
        model = FeatureExtractorWrapper(base_model)
        output_name = "last_hidden_state"
        filename = output_path or "videomae_embeddings.onnx"
        print("🧠 Task: Exporting Raw Embeddings (Feature Extractor)")

    model.eval()

    # --- 4. PREPARE INPUT ---
    # Standard VideoMAE: 16 frames, 3 channels, 224x224
    dummy_input = torch.randn(1, 16, 3, 224, 224)

    # --- 5. EXPORT ---
    print(f"🚀 Exporting to {filename} ...")
    
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=[output_name],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            output_name: {0: 'batch_size'}
        }
    )
    print("✅ Export Complete.")
    print("-" * 30)

# --- RUNNING THE SCRIPT ---
if __name__ == "__main__":
    # You can change this ID to test different models
    
    # CASE A: Base Model (Should export Embeddings)
    export_videomae("MCG-NJU/videomae-base")
    
    print("\n")
    
    # CASE B: Finetuned Model (Should export Classification)
    export_videomae("MCG-NJU/videomae-base-finetuned-kinetics")