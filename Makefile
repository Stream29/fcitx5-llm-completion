CMAKE ?= $(shell command -v cmake 2>/dev/null || ls -1d $(HOME)/Android/Sdk/cmake/*/bin/cmake 2>/dev/null | tail -n 1)
CTEST ?= $(shell command -v ctest 2>/dev/null || ls -1d $(HOME)/Android/Sdk/cmake/*/bin/ctest 2>/dev/null | tail -n 1)
GRADLE ?= $(shell ls -1d $(HOME)/.gradle/wrapper/dists/gradle-9.6.1-bin/*/gradle-9.6.1/bin/gradle 2>/dev/null | tail -n 1 || command -v gradle 2>/dev/null)
JAVA_HOME_ANDROID ?= $(HOME)/.sdkman/candidates/java/17.0.19-tem
ANDROID_SDK_DIR ?= $(HOME)/Android/Sdk
BUILD_DIR := build
FCITX5_LIB_DIR := /usr/lib/x86_64-linux-gnu/fcitx5
FCITX5_USER_ADDON_DIR := $(HOME)/.local/share/fcitx5/addon

.PHONY: all configure build-addon test android-apk dist install uninstall clean

all: build-addon

configure:
	@"$(CMAKE)" -S . -B "$(BUILD_DIR)" -G Ninja -DCMAKE_BUILD_TYPE=Release

build-addon: configure
	@"$(CMAKE)" --build "$(BUILD_DIR)"

test: build-addon
	@"$(CTEST)" --test-dir "$(BUILD_DIR)" --output-on-failure

android-apk:
	JAVA_HOME="$(JAVA_HOME_ANDROID)" ANDROID_HOME="$(ANDROID_SDK_DIR)" \
		"$(GRADLE)" -p android-plugin assembleDebug --no-daemon
	@mkdir -p dist
	@cp android-plugin/build/outputs/apk/debug/*.apk \
		dist/org.fcitx.fcitx5.android.plugin.llm_completion-v2.0.0-arm64-v8a.apk
	@echo "APK ready: dist/org.fcitx.fcitx5.android.plugin.llm_completion-v2.0.0-arm64-v8a.apk"

dist: build-addon
	@mkdir -p dist
	tar -czf dist/fcitx5-llm-completion-addon-linux-x86_64.tar.gz \
		-C "$(BUILD_DIR)" libllmcompletion.so \
		-C ../src/addon llmcompletion.conf

install: build-addon
	mkdir -p "$(FCITX5_USER_ADDON_DIR)"
	install -m 0644 src/addon/llmcompletion.conf "$(FCITX5_USER_ADDON_DIR)/llmcompletion.conf"
	run0 install -o root -g root -m 0755 "$(BUILD_DIR)/libllmcompletion.so" "$(FCITX5_LIB_DIR)/libllmcompletion.so"

uninstall:
	rm -f "$(FCITX5_USER_ADDON_DIR)/llmcompletion.conf"
	run0 rm -f "$(FCITX5_LIB_DIR)/libllmcompletion.so"

clean:
	rm -rf "$(BUILD_DIR)" dist android-plugin/build android-plugin/.cxx android-plugin/.gradle
