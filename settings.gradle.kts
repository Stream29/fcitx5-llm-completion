pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

rootProject.name = "fcitx5-llm-completion"

include(":core")
include(":daemon")
include(":android-plugin")
