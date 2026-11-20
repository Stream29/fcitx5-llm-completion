plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}

allprojects {
    group = "io.github.stream29.fcitx5llm"
    version = "0.1.0"

    repositories {
        mavenCentral()
    }
}
