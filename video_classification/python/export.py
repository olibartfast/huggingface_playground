import torch
import os
from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForVideoClassification,
    VideoMAEImageProcessor
)

# --- 1. THE SMART WRAPPER ---
# This is crucial. It standardizes the output of all these different models
# so ONNX doesn't crash on Dictionaries or Tuples.
class UniversalVideoWrapper(torch.nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        
        # Priority 1: Classification Logic (Logits)
        if hasattr(outputs, "logits"):
            return outputs.logits
            
        # Priority 2: Embeddings Logic (Last Hidden State)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
            
        # Priority 3: Fallback for older models (Tuple output)
        # Some models like Swin might return (last_hidden_state, pooler)
        if isinstance(outputs, tuple):
            return outputs[0]
            
        return outputs

def export_model(model_id):
    print(f"\n" + "="*50)
    print(f"👀 PROCESSING: {model_id}")
    print("="*50)

    # --- 2. ARCHITECTURE DETECTION ---
    try:
        config = AutoConfig.from_pretrained(model_id)
        archs = getattr(config, "architectures", [])
        arch_str = str(archs)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # --- 3. DETERMINE MODE ---
    # We look for "ForVideoClassification" to decide if it has a head.
    # Note: V-JEPA and VideoMAE-Base usually don't have this tag.
    is_classifier = any("ForVideoClassification" in a for a in archs)
    
    # Override for specific edge cases if needed, but auto-detection usually works
    if is_classifier:
        print(f"⚙️  Type detected: CLASSIFICATION [{archs[0]}]")
        hf_model = AutoModelForVideoClassification.from_pretrained(model_id)
        mode = "classification"
        output_name = "logits"
        suffix = "class"
    else:
        print(f"⚙️  Type detected: FEATURE EXTRACTOR / EMBEDDINGS [{archs[0]}]")
        hf_model = AutoModel.from_pretrained(model_id)
        mode = "embeddings"
        output_name = "last_hidden_state"
        suffix = "embed"

    hf_model.eval()
    
    # Wrap it
    model = UniversalVideoWrapper(hf_model, mode)

    # --- 4. PREPARE INPUT ---
    # Standard standard for 99% of video models: (Batch, Frames, Channels, Height, Width)
    # Using 16 frames is the safest default for all these models.
    dummy_input = torch.randn(1, 16, 3, 224, 224)

    # Define filename
    safe_name = model_id.split("/")[-1]
    filename = f"{safe_name}_{suffix}.onnx"

    # --- 5. EXPORT ---
    print(f"🚀 Exporting to {filename} ...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            export_params=True,
            opset_version=14, # 14 is robust for Transformers
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=[output_name],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                output_name: {0: 'batch_size'}
            }
        )
        print(f"✅ SUCCESS! Saved to {os.getcwd()}/{filename}")
        
        # Print expected output shape for user clarity
        if mode == "classification":
            print(f"ℹ️  Output Shape: [Batch, Num_Classes] (e.g., [1, 400])")
        else:
            print(f"ℹ️  Output Shape: [Batch, Patches, Vector_Dim] (e.g., [1, 1568, 768])")
            
    except Exception as e:
        print(f"❌ EXPORT FAILED: {e}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # LIST OF MODELS FROM CHAT HISTORY
    models_to_export = [
        # 1. VideoMAE V1 (Base -> Embeddings)
        "MCG-NJU/videomae-base",
        
        # 2. VideoMAE V1 (Finetuned -> Classification)
        "MCG-NJU/videomae-base-finetuned-kinetics",
        
        # 3. VideoMAE V2 (Base -> Embeddings)
        "MCG-NJU/videomae-v2-base",
        
        # 4. ViViT (Finetuned -> Classification)
        "google/vivit-b-16x2-kinetics400",
        
        # 5. TimeSformer (Finetuned -> Classification)
        "facebook/timesformer-base-finetuned-k400",
        
        # 6. VideoSwin (Finetuned -> Classification)
        "microsoft/videoswin-base-k400",
        
        # 7. V-JEPA (Base -> Embeddings)
        "facebook/vjepa-base",
    ]

    print("Starting Batch Export...")
    for m in models_to_export:
        export_model(m)