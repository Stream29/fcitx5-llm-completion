package org.fcitx.fcitx5.android.plugin.llm_completion;

import android.app.Activity;
import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class SettingsActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int pad = (int) (16f * getResources().getDisplayMetrics().density);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("Fcitx5 RWKV-8 Completion & Collector (v2.0.0)");
        title.setTextSize(19f);
        root.addView(title);

        TextView body = new TextView(this);
        body.setTextSize(14f);
        body.setText(
            "\n• 架构：纯血 C++20 单 libllmcompletion.so（零网络权限、零后台进程）\n"
            + "• 语料收集模式：未放置模型时自动以零功耗 Collector-Only 模式运行，仅在本地记录非密码输入事件流\n"
            + "• 隐私保护：硬件级 CapabilityFlag::Password 自动跳过 + 密码管理器包名内置黑名单 + 工具栏「内联补全: 开/关」随时暂停\n\n"
            + "事件流日志路径：\n"
            + "/storage/emulated/0/Android/data/org.fcitx.fcitx5.android/files/data/llm-completion/predictions.jsonl\n\n"
            + "自定义黑名单（每行一个 Android 包名）：\n"
            + "/storage/emulated/0/Android/data/org.fcitx.fcitx5.android/files/data/llm-completion/blacklist.txt\n\n"
            + "可选本地 NPU 模型目录：\n"
            + "/storage/emulated/0/Android/data/org.fcitx.fcitx5.android/files/data/llm-completion/models/unicode-37m-rwkv8"
        );
        root.addView(body);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);
    }
}
