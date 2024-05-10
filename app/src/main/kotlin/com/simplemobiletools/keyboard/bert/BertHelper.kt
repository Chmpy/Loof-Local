package com.simplemobiletools.keyboard.bert

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.channels.FileChannel

class BertHelper(context: Context) {
    private var vocab: Map<String, Int>
    private var bertInterpreter: Interpreter

    init {
        vocab = loadVocabulary(context)
        bertInterpreter = loadModel(context, vocab)
        Log.d("BertHelper", "BERT interpreter loaded")
    }

    private fun loadVocabulary(context: Context): Map<String, Int> {
        val assetManager = context.assets
        val inputStream = assetManager.open("vocab 2023.json")
        val json = inputStream.bufferedReader().use { it.readText() }
        val mapType = object : TypeToken<Map<String, Int>>() {}.type
        return Gson().fromJson(json, mapType)
    }

    private fun loadModel(context: Context, vocab: Map<String, Int>) : Interpreter {
        val assetFileDescriptor = context.assets.openFd("robbert-2023-dutch-large.tflite")
        val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
        val opts = Interpreter.Options()
        opts.setNumThreads(4)
        return Interpreter(modelBuffer, opts)
    }

    fun runBertInference(input: String): String {
        Log.d("BertHelper", "Running BERT inference on input: $input")
        val preprocessedInput = BertPreprocessor.preprocess(input, vocab)

        // Get input tensor shape and data type
        val inputTensorShape = bertInterpreter.getInputTensor(0).shape()
        Log.d("BertHelper", "Input tensor shape: ${inputTensorShape.contentToString()}")

        // Create input tensor from preprocessed data
        val inputTensor = TensorBuffer.createFixedSize(inputTensorShape, DataType.FLOAT32)
        inputTensor.loadArray(preprocessedInput.tokenIds)

        // Run inference
        val outputTensor = TensorBuffer.createFixedSize(bertInterpreter.getOutputTensor(0).shape(), DataType.FLOAT32)
        val outputBuffer = outputTensor.buffer.rewind()
        bertInterpreter.run(inputTensor.buffer, outputBuffer)

        // Postprocess the output tensor to get the final result
        return BertPostprocessor.postprocess(outputTensor)
    }
}
