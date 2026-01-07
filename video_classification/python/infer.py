import onnxruntime as ort
import numpy as np

def run_inference(onnx_file):
    session = ort.InferenceSession(onnx_file)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Create dummy data
    data = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)
    
    result = session.run([output_name], {input_name: data})[0]
    
    print(f"--- Testing {onnx_file} ---")
    print(f"Output Name: {output_name}")
    print(f"Output Shape: {result.shape}")
    
    if output_name == "logits":
        print("Result: Probabilities for classes")
    elif output_name == "last_hidden_state":
        print("Result: Embeddings (Batch, Patches, Vector_Size)")

# Run on the files created by the script above
run_inference("videomae_embeddings.onnx")
run_inference("videomae_classification.onnx")