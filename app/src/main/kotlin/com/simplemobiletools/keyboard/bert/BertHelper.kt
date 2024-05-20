package com.simplemobiletools.keyboard.bert

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.channels.FileChannel

class BertHelper(context: Context) {
    private var vocab: Map<String, Int>
    private var bertInterpreter: Interpreter

    init {
        vocab = loadVocabulary(context)
        bertInterpreter = loadModel(context)
        Log.d("BertHelper", "BERT interpreter loaded")
    }

    private fun loadVocabulary(context: Context): Map<String, Int> {
        val assetManager = context.assets
        val inputStream = assetManager.open("vocab 2023.json")
        val json = inputStream.bufferedReader().use { it.readText() }
        val mapType = object : TypeToken<Map<String, Int>>() {}.type
        return Gson().fromJson(json, mapType)
    }

    private fun loadModel(context: Context): Interpreter {
        val assetFileDescriptor = context.assets.openFd("robbert_model.tflite")
        val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
        val opts = Interpreter.Options()
        opts.setNumThreads(4)
        val flexDelegate = FlexDelegate()
        opts.addDelegate(flexDelegate)

        return Interpreter(modelBuffer, opts)
    }

    fun runBertInference(input: String): String {
        Log.d("BertHelper", "Running BERT inference on input: $input")

        // Hardcoded values based on the provided information, converted to FLOAT32
        val inputIds = floatArrayOf(0.0f, 21648.0f, 2579.0f, 3.0f)
        val attentionMask = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f,)

        val inputTensorShape = intArrayOf(1, inputIds.size)
        val outputTensorShape = bertInterpreter.getOutputTensor(0).shape()
        val inputDataType = DataType.FLOAT32
        val outputDataType = DataType.FLOAT32

        // Resize the input tensors to match the actual input shapes
        bertInterpreter.resizeInput(0, outputTensorShape, true)
        bertInterpreter.resizeInput(1, outputTensorShape, true)
        bertInterpreter.allocateTensors()

        Log.d("BertHelper", "Input tensor shape: ${inputTensorShape.contentToString()}")
        Log.d("BertHelper", "Output tensor shape: ${outputTensorShape.contentToString()}")
        Log.d("BertHelper", "Input tensor data type: $inputDataType")
        Log.d("BertHelper", "Output tensor data type: $outputDataType")

        // Prepare the input tensors
        val inputIdsTensor = TensorBuffer.createFixedSize(inputTensorShape, inputDataType)
        val attentionMaskTensor = TensorBuffer.createFixedSize(inputTensorShape, inputDataType)
        val outputTensor = TensorBuffer.createFixedSize(outputTensorShape, outputDataType)

        // Load hardcoded data into tensors
        inputIdsTensor.loadArray(inputIds, inputTensorShape)
        attentionMaskTensor.loadArray(attentionMask, inputTensorShape)

        Log.d("BertHelper", "Input tensor data: ${inputIdsTensor.floatArray.contentToString()}")
        Log.d("BertHelper", "Attention mask tensor data: ${attentionMaskTensor.floatArray.contentToString()}")
        Log.d("BertHelper", "Output tensor data: ${outputTensor.floatArray.contentToString()}")

        // Run inference
        try {
            bertInterpreter.run(inputIdsTensor.buffer, outputTensor.buffer)
        } catch (e: Exception) {
            Log.e("BertHelper", "Error running inference: ${e.message}")
            throw e
        }

        // Postprocess the output tensor to get the final result
        return BertPostprocessor.postprocess(outputTensor)
    }


}
