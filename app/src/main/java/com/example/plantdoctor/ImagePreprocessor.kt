package com.example.plantdoctor

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

/**
 * Handles image preprocessing for TensorFlow Lite model inference.
 * Converts Bitmap to UINT8 RGB ByteBuffer with center cropping and resizing.
 */
class ImagePreprocessor {

    companion object {
        private const val INPUT_SIZE = 224
        private const val PIXEL_SIZE = 3 // RGB
        private const val BUFFER_SIZE = INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE
    }

    /**
     * Preprocesses a bitmap for model input.
     * Steps:
     * 1. Center crop to square
     * 2. Resize to 224x224
     * 3. Extract RGB bytes (UINT8, 0-255)
     * 4. Fill ByteBuffer in NHWC layout
     *
     * @param bitmap Input image
     * @return ByteBuffer ready for TFLite inference (UINT8, RGB, NHWC)
     */
    fun preprocess(bitmap: Bitmap): ByteBuffer {
        // Step 1: Center crop to square
        val croppedBitmap = centerCropSquare(bitmap)

        // Step 2: Resize to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(
            croppedBitmap,
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )

        // Step 3: Convert to ByteBuffer
        val byteBuffer = ByteBuffer.allocateDirect(BUFFER_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Step 4: Fill buffer with RGB values (UINT8)
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val value = intValues[pixel++]
                // Extract RGB (no normalization, UINT8 range 0-255)
                byteBuffer.put(((value shr 16) and 0xFF).toByte()) // R
                byteBuffer.put(((value shr 8) and 0xFF).toByte())  // G
                byteBuffer.put((value and 0xFF).toByte())          // B
            }
        }

        // Recycle bitmaps to free memory
        if (croppedBitmap != bitmap) {
            croppedBitmap.recycle()
        }
        if (resizedBitmap != croppedBitmap) {
            resizedBitmap.recycle()
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    /**
     * Center crops the bitmap to a square.
     *
     * @param bitmap Input bitmap
     * @return Square bitmap (center cropped)
     */
    private fun centerCropSquare(bitmap: Bitmap): Bitmap {
        val size = min(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        return Bitmap.createBitmap(bitmap, x, y, size, size)
    }
}
