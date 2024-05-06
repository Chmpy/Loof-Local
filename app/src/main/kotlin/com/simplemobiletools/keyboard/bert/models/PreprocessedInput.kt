package com.simplemobiletools.keyboard.bert.models

data class PreprocessedInput(
    val tokenIds: IntArray,
    val segmentIds: IntArray,
    val attentionMask: IntArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PreprocessedInput

        if (!tokenIds.contentEquals(other.tokenIds)) return false
        if (!segmentIds.contentEquals(other.segmentIds)) return false
        if (!attentionMask.contentEquals(other.attentionMask)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = tokenIds.contentHashCode()
        result = 31 * result + segmentIds.contentHashCode()
        result = 31 * result + attentionMask.contentHashCode()
        return result
    }
}
