package com.example.plantdoctor

/**
 * Data class representing a classification result.
 *
 * @property label The predicted class label (e.g., "Tomato___Late_blight")
 * @property confidence The confidence score in range [0.0, 1.0]
 */
data class Result(
    val label: String,
    val confidence: Float
) {
    /**
     * Returns a formatted confidence percentage string.
     */
    fun getConfidencePercentage(): String {
        return String.format("%.1f%%", confidence * 100)
    }
}
