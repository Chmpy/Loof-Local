package com.simplemobiletools.keyboard.bert

import android.util.Log
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

object BertPostprocessor {
    fun postprocess(outputTensor: TensorBuffer): String {
        Log.d("BertPostprocessor", "Postprocessing output tensor, content: ${outputTensor.floatArray.contentToString()}")
        val outputData = outputTensor.floatArray

        // Check that the output tensor has two values (logits for binary classification)
        require(outputData.size == 2) { "Expected output tensor to have exactly 2 elements, but got ${outputData.size}" }

        // Apply the softmax function to convert logits to probabilities
        val probabilities = softmax(outputData)

        // Determine the predicted class (0 or 1) based on the higher probability
        val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

        // Return the corresponding result based on the predicted class
        return when (predictedClass) {
            0 -> "Negative"
            1 -> "Positive"
            else -> "Unknown"
        }
    }

    // Softmax function to convert logits to probabilities
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: throw IllegalArgumentException("Logits cannot be empty")
        val exps = logits.map { kotlin.math.exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { (it / sumExps).toFloat() }.toFloatArray()
    }
}
