package org.fcitx.fcitx5.android.plugin.llm_completion

import android.app.Activity
import android.content.Context
import android.os.Bundle
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast

class SettingsActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val prefs = getSharedPreferences(LlmCompletionService.PREFS_NAME, Context.MODE_PRIVATE)
        val pad = (16 * resources.displayMetrics.density).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }

        val title = TextView(this).apply {
            text = "Fcitx5 LLM Completion"
            textSize = 20f
        }
        root.addView(title)

        val keyLabel = TextView(this).apply {
            text = "\nAliyun DashScope API Key"
        }
        root.addView(keyLabel)

        val keyInput = EditText(this).apply {
            hint = "sk-..."
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setText(prefs.getString(LlmCompletionService.KEY_API_KEY, ""))
        }
        root.addView(keyInput)

        val blLabel = TextView(this).apply {
            text = "\n应用黑名单（每行一个包名，# 开头为注释）"
        }
        root.addView(blLabel)

        val blInput = EditText(this).apply {
            hint = "com.example.bank"
            minLines = 4
            setText(prefs.getString(LlmCompletionService.KEY_BLACKLIST, ""))
        }
        root.addView(blInput)

        val saveBtn = Button(this).apply {
            text = "保存配置"
            setOnClickListener {
                prefs.edit()
                    .putString(LlmCompletionService.KEY_API_KEY, keyInput.text.toString().trim())
                    .putString(LlmCompletionService.KEY_BLACKLIST, blInput.text.toString())
                    .apply()
                Toast.makeText(this@SettingsActivity, "已保存（无需重启立即生效）", Toast.LENGTH_SHORT).show()
            }
        }
        root.addView(saveBtn)

        val scroll = ScrollView(this).apply { addView(root) }
        setContentView(scroll)
    }
}
