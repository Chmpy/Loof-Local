package com.simplemobiletools.keyboard.bert

import android.util.Log
import com.simplemobiletools.keyboard.bert.models.PreprocessedInput
import com.londogard.nlp.tokenizer.HuggingFaceTokenizerWrapper

object BertPreprocessor {
    private const val MAX_SEQUENCE_LENGTH = 512  // Maximum sequence length for BERT model
    private val tokenizer = HuggingFaceTokenizerWrapper("robbert-v2-dutch-base")  // Example, adjust model name as necessary

    fun preprocess(input: String, vocab: Map<String, Int>): PreprocessedInput {
        var tokens = tokenize(input)
        if (tokens.size > MAX_SEQUENCE_LENGTH - 2) {
            tokens = tokens.take(MAX_SEQUENCE_LENGTH - 2)  // Leave space for special tokens
        }
        val processedTokens = addSpecialTokens(tokens)
        val tokenIds = convertToTokenIds(processedTokens, vocab)
        val segmentIds = createSegmentIds(processedTokens.size)
        val attentionMask = createAttentionMask(processedTokens.size)

        return PreprocessedInput(tokenIds, segmentIds, attentionMask)
    }

    private fun tokenize(input: String): Array<ai.djl.huggingface.tokenizers.Encoding> {
        return tokenizer.batchEncode(input.split(" "))
    }

    private fun addSpecialTokens(tokens: List<String>): List<String> {
        val processedTokens = mutableListOf("[CLS]")
        processedTokens.addAll(tokens)
        processedTokens.add("[SEP]")
        return processedTokens
    }

    private fun convertToTokenIds(tokens: List<String>, vocab: Map<String, Int>): IntArray {
        return tokens.map { token ->
            vocab.getOrDefault(token, vocab["[UNK]"] ?: throw IllegalArgumentException("Unknown token"))
        }.toIntArray().apply { require(size <= MAX_SEQUENCE_LENGTH) }
    }

    private fun createSegmentIds(tokenCount: Int): IntArray = IntArray(tokenCount) { 0 }

    private fun createAttentionMask(tokenCount: Int): IntArray = IntArray(tokenCount) { 1 }
}
