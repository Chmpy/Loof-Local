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
        val assetFileDescriptor = context.assets.openFd("1.tflite")
        val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
        val opts = Interpreter.Options()
        opts.setNumThreads(4)
        return Interpreter(modelBuffer, opts)
    }

    fun runBertInference(input: String): String {
        Log.d("BertHelper", "Running BERT inference on input: $input")
        val preprocessedInput = BertPreprocessor.preprocess(input, vocab)

        // Ensure preprocessed data has correct lengths
        val inputLength = preprocessedInput.tokenIds.size
        val segmentLength = preprocessedInput.segmentIds.size
        val attentionMaskLength = preprocessedInput.attentionMask.size

        if (inputLength != segmentLength || inputLength != attentionMaskLength) {
            throw IllegalArgumentException("Mismatch between lengths of token IDs, segment IDs, and attention mask.")
        }

        // Get output tensor shape to determine the expected input length
        val outputTensorShape = bertInterpreter.getOutputTensor(0).shape()
        val expectedInputLength = outputTensorShape[1]  // Assuming the second dimension is the length

        // Define input tensor shapes based on the expected input length
        val inputShape = intArrayOf(1, expectedInputLength)

        // Initialize input arrays to match the expected length
        val tokenIdsFloatArray = FloatArray(expectedInputLength) { 0f }
        val attentionMaskFloatArray = FloatArray(expectedInputLength) { 0f }
        val segmentIdsFloatArray = FloatArray(expectedInputLength) { 0f }

        // Fill the input arrays with the preprocessed data, handling padding/truncation
        for (i in preprocessedInput.tokenIds.indices) {
            if (i < expectedInputLength) {
                tokenIdsFloatArray[i] = preprocessedInput.tokenIds[i].toFloat()
                attentionMaskFloatArray[i] = preprocessedInput.attentionMask[i].toFloat()
                segmentIdsFloatArray[i] = preprocessedInput.segmentIds[i].toFloat()
            }
        }

        // Create input tensors from preprocessed data
        val inputIdsTensor = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
        val attentionMaskTensor = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
        val segmentIdsTensor = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)

        inputIdsTensor.loadArray(tokenIdsFloatArray)
        attentionMaskTensor.loadArray(attentionMaskFloatArray)
        segmentIdsTensor.loadArray(segmentIdsFloatArray)

        // Log the tensor data to ensure correctness
        Log.d("BertHelper", "Token IDs tensor data: ${tokenIdsFloatArray.contentToString()}")
        Log.d("BertHelper", "Attention Mask tensor data: ${attentionMaskFloatArray.contentToString()}")
        Log.d("BertHelper", "Segment IDs tensor data: ${segmentIdsFloatArray.contentToString()}")

        // Create output tensor
        val outputTensorDataType = bertInterpreter.getOutputTensor(0).dataType()
        Log.d("BertHelper", "Output tensor shape: ${outputTensorShape.contentToString()}")
        Log.d("BertHelper", "Output tensor data type: ${outputTensorDataType}")

        val outputTensor = TensorBuffer.createFixedSize(outputTensorShape, outputTensorDataType)

        // Run inference
        val inputs = arrayOf(inputIdsTensor.buffer, attentionMaskTensor.buffer, segmentIdsTensor.buffer)
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
