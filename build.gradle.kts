plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.android.application) apply false
}

allprojects {
    group = "io.github.stream29.fcitx5llm"
    version = "0.1.1"

    repositories {
        google()
        mavenCentral()
    }
}
