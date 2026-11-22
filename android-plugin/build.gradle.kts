plugins {
    id("com.android.application") version "9.1.0"
}

android {
    namespace = "org.fcitx.fcitx5.android.plugin.llm_completion"
    compileSdk = 36
    ndkVersion = "27.1.12297006"

    defaultConfig {
        applicationId = "org.fcitx.fcitx5.android.plugin.llm_completion"
        minSdk = 26
        targetSdk = 36
        versionCode = 4
        versionName = "2.0.0"

        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                targets("llmcompletion")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DFCITX5_LLM_BUILD_TESTS=OFF",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("../CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        debug {
            // Keep exact package name without .debug suffix so release fcitx5-android discovers it.
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = false
        }
    }

    packaging {
        jniLibs {
            // fcitx5-android dlopen()s plugin .so files directly from nativeLibraryDir,
            // and already provides libc++_shared.so & libFcitx5*.so in its own process.
            useLegacyPackaging = true
            excludes += setOf(
                "**/libc++_shared.so",
                "**/libFcitx5*",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
