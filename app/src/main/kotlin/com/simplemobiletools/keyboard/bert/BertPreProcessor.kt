package com.simplemobiletools.keyboard.bert

import com.simplemobiletools.keyboard.bert.models.PreprocessedInput


/**
 * This object is responsible for preprocessing the input for BERT model.
 * It includes tokenization, adding special tokens, converting tokens to their corresponding IDs,
 * creating segment IDs and attention masks.
 */
object BertPreprocessor {
    private const val MAX_SEQUENCE_LENGTH = 512  // Maximum sequence length for BERT model

    /**
     * Preprocesses the input string for BERT model.
     * @param input The input string to be preprocessed.
     * @return PreprocessedInput object containing tokenIds, segmentIds, and attentionMask.
     */
    fun preprocess(input: String): PreprocessedInput {
        val tokens = tokenize(input)
        val processedTokens = addSpecialTokens(tokens)
        val tokenIds = convertToTokenIds(processedTokens)
        val segmentIds = createSegmentIds(processedTokens.size)
        val attentionMask = createAttentionMask(processedTokens.size)

        return PreprocessedInput(tokenIds, segmentIds, attentionMask)
    }

    /**
     * Tokenizes the input string.
     * @param input The input string to be tokenized.
     * @return List of tokens.
     */
    private fun tokenize(input: String): List<String> {
        return input.split(" ").toList()
    }

    /**
     * Adds special tokens ([CLS] and [SEP]) to the list of tokens.
     * @param tokens The original list of tokens.
     * @return List of tokens with special tokens added.
     */
    private fun addSpecialTokens(tokens: List<String>): List<String> {
        val processedTokens = mutableListOf<String>()
        processedTokens.add("[CLS]")
        processedTokens.addAll(tokens)
        processedTokens.add("[SEP]")
        return processedTokens
    }

    /**
     * Converts tokens to their corresponding IDs using byteEncoder.
     * Throws an exception if the number of tokens exceeds MAX_SEQUENCE_LENGTH.
     * @param tokens The list of tokens to be converted.
     * @return Array of token IDs.
     */
    private fun convertToTokenIds(tokens: List<String>): IntArray {
        return tokens.mapNotNull { token ->
            byteEncoder.entries.find { it.value == token }?.key
        }.toIntArray().apply { require(size <= MAX_SEQUENCE_LENGTH)}
    }

    /**
     * Creates segment IDs for BERT model.
     * @param tokenCount The number of tokens.
     * @return Array of segment IDs.
     */
    private fun createSegmentIds(tokenCount: Int): IntArray {
        return IntArray(tokenCount) { 0 }
    }

    /**
     * Creates attention mask for BERT model.
     * @param tokenCount The number of tokens.
     * @return Array of attention masks.
     */
    private fun createAttentionMask(tokenCount: Int): IntArray {
        return IntArray(tokenCount) { 1 }
    }
}
