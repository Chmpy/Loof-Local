package com.simplemobiletools.keyboard.bert

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

object BertPostprocessor {
    fun postprocess(outputTensor: TensorBuffer): String {
        val outputData = outputTensor.floatArray

        // Assume outputData contains a single float that indicates correctness:
        // 1.0 for correct, 0.0 for incorrect
        val isCorrect = outputData[0] > 0.5  // Assuming threshold at 0.5 for binary classification

        return if (isCorrect) "Ja" else "Nee"
    }
}

