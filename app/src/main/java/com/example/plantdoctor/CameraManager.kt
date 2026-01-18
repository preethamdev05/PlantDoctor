package com.example.plantdoctor

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Manages CameraX lifecycle and image capture operations.
 */
class CameraManager(private val context: Context) {

    private var imageCapture: ImageCapture? = null
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    /**
     * Starts the camera and binds to lifecycle.
     */
    fun startCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        onError: (String) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    // Ensure we always get a compatible format for conversion below
                    .setBufferFormat(ImageFormat.YUV_420_888)
                    .build()

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageCapture
                )
            } catch (e: Exception) {
                onError("Camera initialization failed: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Captures an image and returns it as a Bitmap.
     */
    fun captureImage(
        onImageCaptured: (Bitmap) -> Unit,
        onError: (String) -> Unit
    ) {
        val currentImageCapture = imageCapture
        if (currentImageCapture == null) {
            onError("Camera not ready")
            return
        }

        currentImageCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val bitmap = imageProxyToBitmap(image)
                        image.close()
                        onImageCaptured(bitmap)
                    } catch (e: Exception) {
                        image.close()
                        onError("Image conversion failed: ${e.message}")
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    onError("Image capture failed: ${exception.message}")
                }
            }
        )
    }

    /**
     * Converts ImageProxy (YUV_420_888) to Bitmap safely.
     *
     * Previous code assumed plane buffers are tightly packed; on many devices
     * rowStride/pixelStride causes that to break and produces tiny/invalid JPEG data,
     * leading to errors like "length=1; index=1".
     */
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        if (image.format != ImageFormat.YUV_420_888) {
            throw IllegalArgumentException("Unsupported image format: ${image.format}")
        }

        val nv21 = yuv420888ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        val ok = yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 95, out)
        if (!ok) {
            throw IllegalStateException("YuvImage.compressToJpeg failed")
        }
        val imageBytes = out.toByteArray()
        val bmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: throw IllegalStateException("Bitmap decode failed")
        return bmp
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val nv21 = ByteArray(width * height + (width * height / 2))

        var pos = 0

        // Copy Y plane
        val y = ByteArray(yBuffer.remaining())
        yBuffer.get(y)
        var ySrcIndex = 0
        for (row in 0 until height) {
            System.arraycopy(y, ySrcIndex, nv21, pos, width)
            pos += width
            ySrcIndex += yRowStride
        }

        // Copy UV planes (interleaved VU for NV21)
        val u = ByteArray(uBuffer.remaining())
        val v = ByteArray(vBuffer.remaining())
        uBuffer.get(u)
        vBuffer.get(v)

        var uvSrcIndex = 0
        for (row in 0 until height / 2) {
            var col = 0
            while (col < width) {
                val uvIndex = uvSrcIndex + col * uvPixelStride
                // NV21 expects V then U
                nv21[pos++] = v[uvIndex]
                nv21[pos++] = u[uvIndex]
                col += 2
            }
            uvSrcIndex += uvRowStride
        }

        return nv21
    }

    /**
     * Releases camera resources.
     */
    fun shutdown() {
        cameraExecutor.shutdown()
    }
}
