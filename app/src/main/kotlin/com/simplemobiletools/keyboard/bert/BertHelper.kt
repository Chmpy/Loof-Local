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

    private fun loadModel(context: Context, vocab: Map<String, Int>): Interpreter {
        val assetFileDescriptor = context.assets.openFd("robbert_model.tflite")
        val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
        val opts = Interpreter.Options()
        opts.setNumThreads(4)
        opts.setAllowBufferHandleOutput(true)

        // Add Flex delegate to the interpreter options
        val flexDelegate = FlexDelegate()
        opts.addDelegate(flexDelegate)

        return Interpreter(modelBuffer, opts)
    }

    fun runBertInference(input: String): String {
        Log.d("BertHelper", "Running BERT inference on input: $input")
        val preprocessedInput = BertPreprocessor.preprocess(input, vocab)

        // Ensure preprocessed data has correct lengths
        val inputLength = preprocessedInput.tokenIds.size
        val segmentIdsLength = preprocessedInput.segmentIds.size

        if (inputLength != segmentIdsLength) {
            throw IllegalArgumentException("Mismatch between lengths of token IDs and segment IDs.")
        }

        // Get output tensor shape to determine the expected input length
        val outputTensorShape = bertInterpreter.getOutputTensor(0).shape()
        val expectedInputLength = outputTensorShape[1]  // Assuming the second dimension is the length

        // Define input tensor shapes based on the expected input length
        val inputShape = intArrayOf(1, expectedInputLength)

        // Initialize input arrays to match the expected length
        val tokenIdsFloatArray = FloatArray(expectedInputLength)
        val segmentIdsFloatArray = FloatArray(expectedInputLength)

        // Fill the input arrays with the preprocessed data, handling padding/truncation
        for (i in 0 until expectedInputLength) {
            tokenIdsFloatArray[i] = if (i < preprocessedInput.tokenIds.size) preprocessedInput.tokenIds[i].toFloat() else 0f
            segmentIdsFloatArray[i] = if (i < preprocessedInput.segmentIds.size) preprocessedInput.segmentIds[i].toFloat() else 0f
        }

        // Create input tensors from preprocessed data
        val inputIdsTensor = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
        val segmentIdsTensor = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)

        inputIdsTensor.loadArray(tokenIdsFloatArray)
        segmentIdsTensor.loadArray(segmentIdsFloatArray)

        // Log the tensor data to ensure correctness
        Log.d("BertHelper", "Token IDs tensor data: ${tokenIdsFloatArray.contentToString()}")
        Log.d("BertHelper", "Segment IDs tensor data: ${segmentIdsFloatArray.contentToString()}")

        // Create output tensor
        val outputTensorDataType = bertInterpreter.getOutputTensor(0).dataType()
        Log.d("BertHelper", "Output tensor shape: ${outputTensorShape.contentToString()}")
        Log.d("BertHelper", "Output tensor data type: ${outputTensorDataType}")

        val outputTensor = TensorBuffer.createFixedSize(outputTensorShape, outputTensorDataType)

        // Log expected number of input tensors
        Log.d("BertHelper", "Expected number of input tensors: ${bertInterpreter.inputTensorCount}")

        // Ensure the number of input tensors matches the model's expectations
        if (bertInterpreter.inputTensorCount != 2) {
            throw IllegalArgumentException("Model expects 2 input tensors.")
        }

        // Run inference
        val inputs = arrayOf(inputIdsTensor.buffer, segmentIdsTensor.buffer)
        val outputs = hashMapOf(0 to outputTensor.buffer)

        try {
            bertInterpreter.runForMultipleInputsOutputs(inputs, outputs as Map<Int, Any>)
        } catch (e: Exception) {
            Log.e("BertHelper", "Error running inference: ${e.message}")
            throw e
        }

        // Postprocess the output tensor to get the final result
        return BertPostprocessor.postprocess(outputTensor)
    }


}
