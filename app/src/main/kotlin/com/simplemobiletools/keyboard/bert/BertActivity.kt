package com.simplemobiletools.keyboard.bert

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity


class BertActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val text = intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)
        val bertHelper = BertHelper(this)
        val feedback = bertHelper.runBertInference(text.toString())
        val intent = Intent(Intent.ACTION_PROCESS_TEXT)
        intent.putExtra(Intent.EXTRA_PROCESS_TEXT, feedback)
        setResult(RESULT_OK, intent)
        finish()
    }
}
