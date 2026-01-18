#!/usr/bin/env python3
"""
Model Conversion Script for Android TFLite Compatibility

This script converts a TensorFlow/Keras model to TFLite format that works
with Android's Flex delegate (no float16 internal operations).

Usage:
    python convert_model_android.py --input model.h5 --output plant_doctor_edge_int8.tflite

Or if you have a SavedModel:
    python convert_model_android.py --saved_model ./saved_model_dir --output plant_doctor_edge_int8.tflite
"""

import argparse
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

def create_representative_dataset(input_shape=(1, 224, 224, 3), num_samples=100):
    """
    Create a representative dataset for full integer quantization.
    Adjust input_shape if your model uses different dimensions.
    """
    def representative_dataset():
        for _ in range(num_samples):
            # Generate random data in the range [0, 1]
            data = np.random.rand(*input_shape).astype(np.float32)
            yield [data]
    return representative_dataset

def convert_model(input_path=None, saved_model_path=None, output_path="plant_doctor_edge_int8.tflite", 
                  use_int8=True, input_shape=(1, 224, 224, 3)):
    """
    Convert a model to TFLite format compatible with Android.
    
    Args:
        input_path: Path to .h5 or .keras model file
        saved_model_path: Path to SavedModel directory
        output_path: Output .tflite file path
        use_int8: If True, use full integer quantization (recommended)
        input_shape: Model input shape for representative dataset
    """
    print("Loading model...")
    
    if input_path:
        model = tf.keras.models.load_model(input_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif saved_model_path:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    else:
        raise ValueError("Either input_path or saved_model_path must be provided")
    
    print("Configuring converter...")
    
    # CRITICAL: Do NOT use float16
    # Only use integer quantization or float32
    if use_int8:
        print("Using INT8 quantization (recommended for Android)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Provide representative dataset for full integer quantization
        converter.representative_dataset = create_representative_dataset(input_shape)
        
        # Ensure all operations use integers
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS  # Allow Flex ops if needed
        ]
        
        # Force integer-only inference (input/output remain float32 for compatibility)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        print("Using default float32 (no quantization)")
        # No optimizations - pure float32
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
    
    # Allow custom ops if model uses them
    converter.allow_custom_ops = True
    
    # Convert
    print("Converting model...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nTrying without full integer quantization...")
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print size info
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\n✓ Model converted successfully!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"\nNext steps:")
    print(f"  1. Copy {output_path.name} to app/src/main/assets/")
    print(f"  2. Rebuild the Android app")
    print(f"  3. Test on device")
    
    return tflite_model

def verify_model(model_path):
    """
    Verify the converted model by loading it and checking its specs.
    """
    print(f"\nVerifying model: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nInput tensor:")
    for detail in input_details:
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Name: {detail['name']}")
    
    print("\nOutput tensor:")
    for detail in output_details:
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Name: {detail['name']}")
    
    # Test inference
    print("\nTesting inference...")
    input_shape = input_details[0]['shape']
    input_data = np.random.rand(*input_shape).astype(input_details[0]['dtype'])
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✓ Inference successful! Output shape: {output_data.shape}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to Android-compatible TFLite")
    parser.add_argument('--input', type=str, help='Input .h5 or .keras model file')
    parser.add_argument('--saved_model', type=str, help='SavedModel directory')
    parser.add_argument('--output', type=str, default='plant_doctor_edge_int8.tflite',
                        help='Output .tflite file')
    parser.add_argument('--no-int8', action='store_true', 
                        help='Skip INT8 quantization (use float32)')
    parser.add_argument('--input-height', type=int, default=224, help='Input height')
    parser.add_argument('--input-width', type=int, default=224, help='Input width')
    parser.add_argument('--verify', action='store_true', help='Verify the model after conversion')
    
    args = parser.parse_args()
    
    if not args.input and not args.saved_model:
        print("Error: Either --input or --saved_model must be provided")
        parser.print_help()
        sys.exit(1)
    
    input_shape = (1, args.input_height, args.input_width, 3)
    
    try:
        convert_model(
            input_path=args.input,
            saved_model_path=args.saved_model,
            output_path=args.output,
            use_int8=not args.no_int8,
            input_shape=input_shape
        )
        
        if args.verify:
            verify_model(args.output)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
