#!/usr/bin/env python3
"""
Example usage of Qwen2.5-VL Object Detection

This script demonstrates how to use the qwen2.5-vl_object_detection.py module
for object detection tasks.

Before running, make sure to install the required dependencies:
pip install -r requirements.object_detection.txt

Example usage:
    python3 object_detection_example.py
"""

from qwen2_5_vl_object_detection import (
    Qwen2_5_VLObjectDetector, 
    load_image, 
    display_image, 
    save_image, 
    save_detections,
    display_results
)


def example_basic_detection():
    """Basic object detection example"""
    print("="*60)
    print("BASIC OBJECT DETECTION EXAMPLE")
    print("="*60)
    
    # Load image from URL
    image_url = "https://learnopencv.com/wp-content/uploads/2025/06/elephants.jpg"
    image = load_image(image_url)
    print("Image loaded successfully!")
    
    # Initialize detector (uses 7B model by default)
    detector = Qwen2_5_VLObjectDetector()
    
    # Run object detection
    detections = detector.detect_objects(image, "Detect all animals in this image")
    
    # Display results
    display_results("Detect all animals in this image", detections)
    
    return detections, image


def example_custom_prompts():
    """Example with different detection prompts"""
    print("\n" + "="*60)
    print("CUSTOM PROMPTS EXAMPLE")
    print("="*60)
    
    # Load image
    image_url = "https://learnopencv.com/wp-content/uploads/2025/06/elephants.jpg"
    image = load_image(image_url)
    
    # Initialize detector
    detector = Qwen2_5_VLObjectDetector()
    
    # Different prompts for different detection tasks
    prompts = [
        "Detect elephants in this image",
        "Find all large animals",
        "Identify the main subjects in this photo",
        "Locate all living creatures"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: {prompt} ---")
        detections = detector.detect_objects(image, prompt)
        print(f"Found {len(detections)} objects")
        
        for j, det in enumerate(detections, 1):
            label = det.get("label", "unknown")
            bbox = det.get("bbox_2d", [])
            print(f"  {j}. {label} at {bbox}")


def example_with_visualization():
    """Example that includes visualization and saving results"""
    print("\n" + "="*60)
    print("VISUALIZATION AND SAVING EXAMPLE")
    print("="*60)
    
    # Load image
    image_url = "https://learnopencv.com/wp-content/uploads/2025/06/elephants.jpg"
    image = load_image(image_url)
    
    # Initialize detector
    detector = Qwen2_5_VLObjectDetector()
    
    # Run detection with visualization
    detections, annotated_image = detector.detect_and_visualize(
        image, 
        "Outline the position of elephants",
        max_new_tokens=1000,
        box_color="lime",
        box_width=4,
        font_size=36,
        text_color="black",
        text_bg="yellow"
    )
    
    # Display results
    display_results("Outline the position of elephants", detections)
    
    # Save results (optional - uncomment to save)
    # save_image(annotated_image, "output_elephants_detected.jpg")
    # save_detections(detections, "elephant_detections.json")
    
    print("Visualization complete! Use display_image() to show the results.")
    return detections, annotated_image


def example_local_image():
    """Example using a local image file"""
    print("\n" + "="*60)
    print("LOCAL IMAGE EXAMPLE")
    print("="*60)
    
    # This would work with a local image file
    # image_path = "path/to/your/image.jpg"
    # image = load_image(image_path)
    
    # For demo, we'll use the URL example
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = load_image(image_url)
    
    # Initialize detector with 3B model (lighter)
    detector = Qwen2_5_VLObjectDetector("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Run detection
    detections = detector.detect_objects(image, "What objects can you see in this image?")
    
    # Display results
    display_results("What objects can you see in this image?", detections)
    
    return detections, image


def main():
    """Run all examples"""
    print("Qwen2.5-VL Object Detection Examples")
    print("Note: These examples require GPU and the model downloads (several GB)")
    print("The first run will download the model, which may take time.\n")
    
    try:
        # Example 1: Basic detection
        detections1, image1 = example_basic_detection()
        
        # Example 2: Custom prompts
        example_custom_prompts()
        
        # Example 3: With visualization
        detections3, annotated_image3 = example_with_visualization()
        
        # Example 4: Local image
        detections4, image4 = example_local_image()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("1. Installed all requirements: pip install -r requirements.object_detection.txt")
        print("2. A GPU with sufficient VRAM (7GB+ for 7B model, 3GB+ for 3B model)")
        print("3. Internet connection for downloading models and images")


if __name__ == "__main__":
    main()
