#!/usr/bin/env python3
"""
Example usage of the enhanced Qwen2.5-VL script with multi-image support
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show the description"""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

def main():
    script_path = "qwen2.5-vl.py"
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found in current directory")
        return 1
    
    print("🖼️ Qwen2.5-VL Multi-Image Examples")
    print("This script demonstrates various ways to use the enhanced Qwen2.5-VL script")
    
    # Example 1: Single image (original functionality)
    run_command([
        "python3", script_path,
        "--prompt", "Describe this image in detail.",
        "--max_new_tokens", "150"
    ], "Single image analysis")
    
    # Example 2: Multiple images from URLs
    run_command([
        "python3", script_path,
        "--image_urls",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=400",
        "--prompt", "Compare these two images. What are the similarities and differences?",
        "--max_new_tokens", "200"
    ], "Multiple images from URLs - comparison task")
    
    # Example 3: Multiple images with different prompts
    run_command([
        "python3", script_path,
        "--image_urls",
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
        "--prompt", "I have uploaded multiple dog images. Describe the breed, pose, and setting of each dog.",
        "--max_new_tokens", "250",
        "--visualize"
    ], "Multiple dog images - detailed analysis with visualization")
    
    # Example 4: Using pipeline approach
    run_command([
        "python3", script_path,
        "--image_urls",
        "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=400",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
        "--prompt", "What activities or scenes are depicted in these images?",
        "--use_pipeline",
        "--max_new_tokens", "180"
    ], "Multiple images using pipeline approach")
    
    # Example 5: Using Qwen class approach
    run_command([
        "python3", script_path,
        "--image_urls",
        "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400",
        "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400",
        "--prompt", "Analyze the architectural styles shown in these images.",
        "--use_qwen_class",
        "--max_new_tokens", "200"
    ], "Multiple images using Qwen2_5_VLProcessor class")
    
    print(f"\n{'='*60}")
    print("✅ All examples completed!")
    print("\nYou can also use local image paths:")
    print("python3 qwen2.5-vl.py --image_paths img1.jpg img2.jpg --prompt 'Your prompt here'")
    print("\nFor more options, run:")
    print("python3 qwen2.5-vl.py --help")

if __name__ == "__main__":
    main()
