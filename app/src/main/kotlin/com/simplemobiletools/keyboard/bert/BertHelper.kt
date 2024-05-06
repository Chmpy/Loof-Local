package com.simplemobiletools.keyboard.bert

import android.content.Context
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.channels.FileChannel

class BertHelper(context: Context) {
    private var bertInterpreter: Interpreter

    init {
        bertInterpreter = loadModel(context)
    }

    private fun loadModel(context: Context): Interpreter {
        val assetFileDescriptor = context.assets.openFd("robbert-v2-dutch-base.tflite")
        val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
        val opts = Interpreter.Options()
        opts.setNumThreads(4)
        return Interpreter(modelBuffer, opts)
    }


    fun runBertInference(input: String): String {
        Log.d("BertHelper", "Running BERT inference on input: $input")
        val preprocessedInput = BertPreprocessor.preprocess(input)
        val inputTensor = TensorBuffer.createFixedSize(preprocessedInput.tokenIds, DataType.FLOAT32)

        // Run inference
        val outputTensor = TensorBuffer.createFixedSize(preprocessedInput.tokenIds, DataType.FLOAT32)
        bertInterpreter.run(inputTensor.buffer, outputTensor.buffer.rewind())

        // Postprocess the output tensor to get the final result
        return BertPostprocessor.postprocess(outputTensor)
    }
}
