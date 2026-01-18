# CRITICAL: Model Conversion Issue

## Root Cause Analysis

The app crashes with:
```
Inference failed (Input: shape=[1, 224, 224, 3] type=UINT8 bytes=150528 detected_bpe=1): 
Internal error: Failed to run on the given Interpreter: 
Expected tensor of type half but got type float
Node number 796 (TfLiteFlexDelegate) failed to invoke.
```

### What This Means

1. **Input preprocessing is CORRECT**: The model expects UINT8 input (1 byte per element), and the app provides exactly that.

2. **The problem is INTERNAL to the model**: The model file `plant_doctor_edge_int8.tflite` contains operations that use **float16 (half precision)** internally, but the TFLite Flex Delegate on Android **cannot execute float16 operations properly**.

3. **Why it fails**: When the Flex delegate tries to run node 796, it expects float32 tensors but the model specifies float16, causing a type mismatch crash.

## Solution: Reconvert the Model

The model must be reconverted from the original TensorFlow/Keras model with these settings:

```python
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('your_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# CRITICAL: Do NOT use float16 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use INT8 quantization only (not float16)
converter.target_spec.supported_types = [tf.int8]  # NOT tf.float16

# Allow Select TF ops if needed
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter.allow_custom_ops = True

# If you need a representative dataset for full integer quantization:
def representative_dataset():
    for _ in range(100):
        # Provide sample input data matching your model's input shape
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# Convert
tflite_model = converter.convert()

# Save
with open('plant_doctor_edge_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Key Points

- **Remove all float16 references**: Do NOT use `target_spec.supported_types = [tf.float16]`
- **Stick to INT8 or FLOAT32**: Android TFLite with Flex delegate handles these reliably
- **The app code is correct**: No changes needed to the Kotlin/Java code

## Model File Issue

The current `plant_doctor_edge_int8.tflite` in the repo is only 156 bytes - this is a Git LFS pointer file, not the actual model. The actual model needs to be:

1. Reconverted (as above)
2. Properly uploaded to GitHub (or use Git LFS correctly)
3. Tested locally first

## Temporary Workaround (if you can't reconvert)

If you MUST use the current model and it truly requires float16, you would need to:

1. Build a custom TFLite library with proper float16 support
2. OR use TFLite GPU delegate (if your model is GPU-compatible)
3. OR switch to a different inference runtime (TensorFlow Lite Task Library, MediaPipe, etc.)

But the BEST solution is to reconvert the model without float16.
