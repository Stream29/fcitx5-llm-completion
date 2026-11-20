CXX ?= g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -fPIC $(shell pkg-config --cflags Fcitx5Core Fcitx5Utils)
LDFLAGS := -shared $(shell pkg-config --libs Fcitx5Core Fcitx5Utils)

FCITX5_LIB_DIR := /usr/lib/x86_64-linux-gnu/fcitx5
FCITX5_USER_ADDON_DIR := $(HOME)/.local/share/fcitx5/addon
DAEMON_BIN := daemon/build/bin/linuxX64/releaseExecutable/fcitx5-llm-completion-daemon.kexe
USER_BIN_DIR := $(HOME)/.local/bin

.PHONY: all build-addon build-daemon test dist install uninstall clean

all: build-addon build-daemon

build-addon: build/libllmcompletion.so

build/libllmcompletion.so: addon/llmcompletion.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

build-daemon:
	./gradlew :daemon:linkReleaseExecutableLinuxX64

test:
	./gradlew :core:allTests

dist: all
	@mkdir -p dist
	cp "$(DAEMON_BIN)" dist/fcitx5-llm-completion-daemon-linux-x86_64
	chmod 0755 dist/fcitx5-llm-completion-daemon-linux-x86_64
	tar -czf dist/fcitx5-llm-completion-addon-linux-x86_64.tar.gz \
		-C build libllmcompletion.so \
		-C ../addon llmcompletion.conf

install: all
	mkdir -p "$(FCITX5_USER_ADDON_DIR)" "$(USER_BIN_DIR)"
	install -m 0644 addon/llmcompletion.conf "$(FCITX5_USER_ADDON_DIR)/llmcompletion.conf"
	install -m 0755 "$(DAEMON_BIN)" "$(USER_BIN_DIR)/fcitx5-llm-completion-daemon"
	run0 install -m 0755 build/libllmcompletion.so "$(FCITX5_LIB_DIR)/libllmcompletion.so"

uninstall:
	rm -f "$(FCITX5_USER_ADDON_DIR)/llmcompletion.conf"
	rm -f "$(USER_BIN_DIR)/fcitx5-llm-completion-daemon"
	run0 rm -f "$(FCITX5_LIB_DIR)/libllmcompletion.so"

clean:
	rm -rf build dist
