package com.example.plantdoctor

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min

/**
 * TensorFlow Lite classifier for plant disease detection.
 *
 * NOTE: TFLite Java DataType enum does not expose FLOAT16 on some versions.
 * To support float16 ("half") tensors anyway, this code detects float16 via
 * tensor.numBytes() / elementCount == 2, and then feeds/reads FP16 ByteBuffers.
 */
class TFLiteClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    // Derived from model input tensor
    private var inputWidth = 224
    private var inputHeight = 224
    private var inputChannels = 3
    private var inputType: DataType = DataType.UINT8
    private var inputBytesPerElement: Int = 1
    
    // Debug info
    private var debugInfo = ""

    companion object {
        private const val MODEL_FILE = "plant_doctor_edge_int8.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val NUM_THREADS = 4
    }

    @Throws(Exception::class)
    fun initialize() {
        try {
            val modelBuffer = loadModelFile()
            // Add Flex Delegate support implicitly via dependency, but ensure options correct
            val options = Interpreter.Options().apply { 
                setNumThreads(NUM_THREADS)
                // If the model uses Flex ops, we should NOT set strict CPU if it needs GPU/Flex
            }
            interpreter = Interpreter(modelBuffer, options)

            labels = loadLabels()
            if (labels.isEmpty()) throw IllegalStateException("Labels file is empty")

            val t = requireInterpreter().getInputTensor(0)
            readInputSpec(t)
            
            // Generate debug string
            debugInfo = "Input: shape=${t.shape().contentToString()} type=${t.dataType()} bytes=${t.numBytes()} detected_bpe=$inputBytesPerElement"
            
        } catch (e: Exception) {
            close()
            throw Exception("Failed to initialize classifier: ${e.message}", e)
        }
    }

    @Throws(Exception::class)
    fun classify(bitmap: Bitmap): Result {
        val interp = requireInterpreter()

        try {
            val input = preprocess(bitmap)

            val outTensor = interp.getOutputTensor(0)
            val outCount = elementCount(outTensor)

            return when (outTensor.dataType()) {
                DataType.UINT8 -> {
                    val out = ByteArray(outCount)
                    interp.run(input, out)
                    val scores = out.map { it.toInt() and 0xFF }
                    argmaxScores(scores, 255f)
                }

                DataType.FLOAT32 -> {
                    val outBytesPerElement = bytesPerElement(outTensor, outCount)
                    if (outBytesPerElement == 2) {
                        // FLOAT16 output hidden behind FLOAT32 enum in some builds
                        val out = ByteBuffer.allocateDirect(outCount * 2).order(ByteOrder.nativeOrder())
                        interp.run(input, out)
                        out.rewind()
                        val scores = FloatArray(outCount)
                        for (i in 0 until outCount) {
                            scores[i] = halfToFloat(out.short)
                        }
                        argmaxScores(scores.toList(), 1f)
                    } else {
                        val out = FloatArray(outCount)
                        interp.run(input, out)
                        argmaxScores(out.toList(), 1f)
                    }
                }
                
                // If library actually supports FLOAT16 enum in future
                // DataType.FLOAT16 -> { ... }

                else -> throw IllegalStateException("Unsupported output tensor type: ${outTensor.dataType()}")
            }
        } catch (e: Exception) {
            // Include debug info in error message
            throw Exception("Inference failed ($debugInfo): ${e.message}", e)
        }
    }

    private fun argmaxScores(scores: List<Number>, denom: Float): Result {
        val size = min(scores.size, labels.size)
        var maxIndex = 0
        var maxVal = Float.NEGATIVE_INFINITY

        for (i in 0 until size) {
            val v = scores[i].toFloat()
            if (v > maxVal) {
                maxVal = v
                maxIndex = i
            }
        }

        val confidence = if (denom == 1f) maxVal else (maxVal / denom)
        return Result(label = labels[maxIndex], confidence = confidence)
    }

    /**
     * Preprocess bitmap according to model input shape and bytes/element.
     */
    private fun preprocess(bitmap: Bitmap): Any {
        val cropped = centerCropSquare(bitmap)
        val resized = Bitmap.createScaledBitmap(cropped, inputWidth, inputHeight, true)

        val intValues = IntArray(inputWidth * inputHeight)
        resized.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val elemCount = inputWidth * inputHeight * inputChannels

        // Determine target based on detected input bytes per element (2 = FP16, 4 = FP32, 1 = UINT8)
        val result: Any = if (inputBytesPerElement == 2) {
             // Force FP16 path regardless of Enum (often reports FLOAT32 but is actually FLOAT16)
             val buf = ByteBuffer.allocateDirect(elemCount * 2).order(ByteOrder.nativeOrder())
             var p = 0
             for (i in 0 until inputHeight) {
                 for (j in 0 until inputWidth) {
                     val v = intValues[p++]
                     buf.putShort(floatToHalf(((v shr 16) and 0xFF) / 255f))
                     buf.putShort(floatToHalf(((v shr 8) and 0xFF) / 255f))
                     buf.putShort(floatToHalf((v and 0xFF) / 255f))
                 }
             }
             buf.rewind()
             buf
        } else {
            when (inputType) {
                DataType.UINT8 -> {
                    val buf = ByteBuffer.allocateDirect(elemCount)
                    buf.order(ByteOrder.nativeOrder())
                    var p = 0
                    for (i in 0 until inputHeight) {
                        for (j in 0 until inputWidth) {
                            val v = intValues[p++]
                            buf.put(((v shr 16) and 0xFF).toByte())
                            buf.put(((v shr 8) and 0xFF).toByte())
                            buf.put((v and 0xFF).toByte())
                        }
                    }
                    buf.rewind()
                    buf
                }

                DataType.FLOAT32 -> {
                    // Standard FP32
                    val buf = ByteBuffer.allocateDirect(elemCount * 4).order(ByteOrder.nativeOrder())
                    var p = 0
                    for (i in 0 until inputHeight) {
                        for (j in 0 until inputWidth) {
                            val v = intValues[p++]
                            buf.putFloat(((v shr 16) and 0xFF) / 255f)
                            buf.putFloat(((v shr 8) and 0xFF) / 255f)
                            buf.putFloat((v and 0xFF) / 255f)
                        }
                    }
                    buf.rewind()
                    buf
                }

                else -> throw IllegalStateException("Unsupported input tensor type: $inputType")
            }
        }

        if (cropped != bitmap && !cropped.isRecycled) cropped.recycle()
        if (resized != cropped && !resized.isRecycled) resized.recycle()

        return result
    }

    private fun centerCropSquare(bitmap: Bitmap): Bitmap {
        val size = min(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        return Bitmap.createBitmap(bitmap, x, y, size, size)
    }

    private fun readInputSpec(t: Tensor) {
        inputType = t.dataType()
        val shape = t.shape()
        // Expected [1, H, W, C]
        if (shape.size >= 4) {
            inputHeight = shape[1]
            inputWidth = shape[2]
            inputChannels = shape[3]
        }

        val count = elementCount(t)
        inputBytesPerElement = bytesPerElement(t, count)
    }

    private fun elementCount(t: Tensor): Int {
        return t.shape().fold(1) { acc, v -> acc * v }
    }

    private fun bytesPerElement(t: Tensor, elementCount: Int): Int {
        val total = t.numBytes()
        return if (elementCount <= 0) 0 else (total / elementCount)
    }

    private fun requireInterpreter(): Interpreter {
        return interpreter ?: throw IllegalStateException("Classifier not initialized")
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    private fun loadLabels(): List<String> {
        return context.assets.open(LABELS_FILE).bufferedReader().use { reader ->
            reader.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    private fun floatToHalf(f: Float): Short {
        val bits = java.lang.Float.floatToIntBits(f)
        val sign = (bits ushr 16) and 0x8000
        var valExp = (bits ushr 23) and 0xFF
        var valMant = bits and 0x7FFFFF

        if (valExp == 255) {
            return (sign or 0x7C00 or (if (valMant != 0) 0x01 else 0)).toShort()
        }

        valExp = valExp - 127 + 15
        if (valExp >= 31) return (sign or 0x7C00).toShort()
        if (valExp <= 0) {
            if (valExp < -10) return sign.toShort()
            valMant = (valMant or 0x800000) shr (1 - valExp)
            return (sign or ((valMant + 0x1000) shr 13)).toShort()
        }

        return (sign or (valExp shl 10) or ((valMant + 0x1000) shr 13)).toShort()
    }

    private fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xFFFF
        val sign = (bits and 0x8000) shl 16
        var exp = (bits ushr 10) and 0x1F
        var mant = bits and 0x03FF

        val outBits = when {
            exp == 0 -> {
                if (mant == 0) sign
                else {
                    while ((mant and 0x0400) == 0) {
                        mant = mant shl 1
                        exp -= 1
                    }
                    exp += 1
                    mant = mant and 0x03FF
                    sign or ((exp + (127 - 15)) shl 23) or (mant shl 13)
                }
            }
            exp == 31 -> sign or 0x7F800000 or (mant shl 13)
            else -> sign or ((exp + (127 - 15)) shl 23) or (mant shl 13)
        }

        return java.lang.Float.intBitsToFloat(outBits)
    }
}
