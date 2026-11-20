plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.kotlin.serialization)
}

kotlin {
    linuxX64 {
        binaries {
            executable {
                baseName = "fcitx5-llm-completion-daemon"
                entryPoint = "io.github.stream29.fcitx5llm.daemon.main"
            }
        }
    }

    sourceSets {
        linuxX64Main.dependencies {
            implementation(project(":core"))
            implementation(libs.ktor.client.curl)
        }
    }
}
