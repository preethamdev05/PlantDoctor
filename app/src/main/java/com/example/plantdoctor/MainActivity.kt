package com.example.plantdoctor

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Main activity for Plant Doctor app.
 * Handles camera permissions, preview, capture, and ML inference.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var captureButton: FloatingActionButton
    private lateinit var labelText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var errorText: TextView
    private lateinit var captureOverlay: View

    private lateinit var cameraManager: CameraManager
    private lateinit var classifier: TFLiteClassifier

    private var isProcessing = false

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            initializeCamera()
        } else {
            showError(getString(R.string.camera_permission_denied))
            captureButton.isEnabled = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        previewView = findViewById(R.id.previewView)
        captureButton = findViewById(R.id.captureButton)
        labelText = findViewById(R.id.labelText)
        confidenceText = findViewById(R.id.confidenceText)
        errorText = findViewById(R.id.errorText)
        captureOverlay = findViewById(R.id.captureOverlay)

        // Initialize managers
        cameraManager = CameraManager(this)
        classifier = TFLiteClassifier(this)

        // Set up capture button
        captureButton.setOnClickListener {
            if (!isProcessing) {
                captureAndClassify()
            }
        }

        // Initialize classifier
        initializeClassifier()

        // Request camera permission
        checkCameraPermission()
    }

    /**
     * Initializes the TFLite classifier on a background thread.
     */
    private fun initializeClassifier() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                classifier.initialize()
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    val errorMessage = "Model Error: ${e.message}"
                    showError(errorMessage)
                    captureButton.isEnabled = false
                    android.util.Log.e("PlantDoctor", "Classifier initialization failed", e)
                }
            }
        }
    }

    /**
     * Checks and requests camera permission if needed.
     */
    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                initializeCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    /**
     * Initializes the camera.
     */
    private fun initializeCamera() {
        cameraManager.startCamera(
            lifecycleOwner = this,
            previewView = previewView,
            onError = { error ->
                showError(error)
                captureButton.isEnabled = false
            }
        )
    }

    /**
     * Captures an image and runs classification.
     */
    private fun captureAndClassify() {
        isProcessing = true
        captureButton.isEnabled = false
        showOverlay()
        clearError()

        cameraManager.captureImage(
            onImageCaptured = { bitmap ->
                // CameraX callbacks run on a background thread.
                // We proceed to classification, but UI updates must be handled carefully.
                classifyImage(bitmap)
            },
            onError = { error ->
                // FIX: Ensure UI updates run on Main Thread
                runOnUiThread {
                    hideOverlay()
                    showError(error)
                    isProcessing = false
                    captureButton.isEnabled = true
                }
            }
        )
    }

    /**
     * Classifies the captured image on a background thread.
     */
    private fun classifyImage(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val result = classifier.classify(bitmap)

                withContext(Dispatchers.Main) {
                    hideOverlay()
                    displayResult(result)
                    isProcessing = false
                    captureButton.isEnabled = true
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    hideOverlay()
                    showError("${getString(R.string.inference_error)}: ${e.message}")
                    isProcessing = false
                    captureButton.isEnabled = true
                }
            } finally {
                bitmap.recycle()
            }
        }
    }

    /**
     * Displays classification result in UI.
     */
    private fun displayResult(result: Result) {
        labelText.text = formatLabel(result.label)
        confidenceText.text = "Confidence: ${result.getConfidencePercentage()}"
    }

    /**
     * Formats label for display (converts underscores to spaces).
     */
    private fun formatLabel(label: String): String {
        return label.replace("___", " - ").replace("_", " ")
    }

    /**
     * Shows error message.
     */
    private fun showError(message: String) {
        errorText.text = message
        errorText.visibility = View.VISIBLE
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    /**
     * Clears error message.
     */
    private fun clearError() {
        errorText.visibility = View.GONE
        errorText.text = ""
    }

    /**
     * Shows capture overlay.
     */
    private fun showOverlay() {
        captureOverlay.visibility = View.VISIBLE
    }

    /**
     * Hides capture overlay.
     */
    private fun hideOverlay() {
        captureOverlay.visibility = View.GONE
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
        classifier.close()
    }
}
