package com.example.plantdoctor

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TensorFlow Lite classifier for plant disease detection.
 * Uses INT8 quantized model with UINT8 input/output.
 */
class TFLiteClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private val preprocessor = ImagePreprocessor()

    companion object {
        private const val MODEL_FILE = "plant_doctor_edge_int8.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val NUM_THREADS = 4
        private const val OUTPUT_SIZE = 38 // Number of classes
    }

    /**
     * Initializes the TFLite interpreter and loads labels.
     * Must be called before classify().
     *
     * @throws Exception if model or labels cannot be loaded
     */
    @Throws(Exception::class)
    fun initialize() {
        try {
            // Load model
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)
            }
            interpreter = Interpreter(modelBuffer, options)

            // Load labels
            labels = loadLabels()

            if (labels.size != OUTPUT_SIZE) {
                throw IllegalStateException(
                    "Expected $OUTPUT_SIZE labels, found ${labels.size}"
                )
            }
        } catch (e: Exception) {
            close()
            throw Exception("Failed to initialize classifier: ${e.message}", e)
        }
    }

    /**
     * Classifies a plant leaf image.
     *
     * @param bitmap Input image
     * @return Classification result with label and confidence
     * @throws Exception if inference fails
     */
    @Throws(Exception::class)
    fun classify(bitmap: Bitmap): Result {
        val currentInterpreter = interpreter
            ?: throw IllegalStateException("Classifier not initialized")

        try {
            // Preprocess image
            val inputBuffer = preprocessor.preprocess(bitmap)

            // Prepare output buffer (UINT8, shape: 1x38)
            val outputBuffer = ByteBuffer.allocateDirect(OUTPUT_SIZE)
            outputBuffer.rewind()

            // Run inference
            currentInterpreter.run(inputBuffer, outputBuffer)

            // Post-process output
            outputBuffer.rewind()
            val outputArray = ByteArray(OUTPUT_SIZE)
            outputBuffer.get(outputArray)

            // Convert UINT8 to int (0-255) and find argmax
            val scores = outputArray.map { it.toInt() and 0xFF }
            val maxIndex = scores.indices.maxByOrNull { scores[it] } ?: 0
            val maxScore = scores[maxIndex]

            // Calculate confidence (UINT8 to float)
            val confidence = maxScore / 255.0f

            return Result(
                label = labels[maxIndex],
                confidence = confidence
            )
        } catch (e: Exception) {
            throw Exception("Inference failed: ${e.message}", e)
        }
    }

    /**
     * Loads the TFLite model from assets.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Loads labels from assets.
     */
    private fun loadLabels(): List<String> {
        return context.assets.open(LABELS_FILE).bufferedReader().use { reader ->
            reader.lineSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .toList()
        }
    }

    /**
     * Releases resources.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
