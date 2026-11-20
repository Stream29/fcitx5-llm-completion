plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "org.fcitx.fcitx5.android.plugin.llm_completion"
    compileSdk = 36

    defaultConfig {
        applicationId = "org.fcitx.fcitx5.android.plugin.llm_completion"
        minSdk = 23
        targetSdk = 36
        versionCode = 2
        versionName = "0.1.1"
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
            // fcitx5-android dlopen()s plugin .so files directly from nativeLibraryDir.
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":core"))
    implementation(libs.ktor.client.okhttp)
}
