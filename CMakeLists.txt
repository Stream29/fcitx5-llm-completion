cmake_minimum_required(VERSION 3.22)
project(fcitx5_llm_completion VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FCITX5_LLM_BUILD_ADDON "Build Fcitx5 libllmcompletion.so addon" ON)
option(FCITX5_LLM_BUILD_TESTS "Build native C++ unit and NPU tests" ON)

find_package(ZLIB REQUIRED)

set(FCITX5_LLM_CORE_SOURCES
    src/core/npz_reader.cpp
    src/core/unicode_codec.cpp
    src/core/online_sam.cpp
    src/core/online_adapter.cpp
    src/core/suffix_index.cpp
    src/core/text_ops.cpp
    src/core/rwkv8_engine.cpp
)

if(ANDROID)
    list(APPEND FCITX5_LLM_CORE_SOURCES src/backends/android_npu_runner.cpp)
else()
    if(NOT DEFINED OpenVINO_DIR)
        set(_DEFAULT_OV_CMAKE "$ENV{HOME}/.local/share/fcitx5-llm-completion/openvino-cpp/openvino/cmake")
        if(EXISTS "${_DEFAULT_OV_CMAKE}/OpenVINOConfig.cmake")
            set(OpenVINO_DIR "${_DEFAULT_OV_CMAKE}")
        endif()
    endif()
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    list(APPEND FCITX5_LLM_CORE_SOURCES src/backends/openvino_runner.cpp)
endif()

add_library(fcitx5_llm_core STATIC ${FCITX5_LLM_CORE_SOURCES})
target_include_directories(fcitx5_llm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(fcitx5_llm_core PRIVATE -Wall -Wextra -O3)
if(ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(fcitx5_llm_core PRIVATE -march=armv8.2-a+dotprod+fp16)
endif()
target_link_libraries(fcitx5_llm_core PUBLIC ZLIB::ZLIB)

set(_OV_LIB_RPATH "")
if(ANDROID)
    target_link_libraries(fcitx5_llm_core PUBLIC dl log)
else()
    target_link_libraries(fcitx5_llm_core PUBLIC openvino::runtime pthread)
    get_filename_component(_OV_LIB_RPATH "${OpenVINO_DIR}/../libs" REALPATH)
endif()

if(FCITX5_LLM_BUILD_ADDON)
    if(ANDROID)
        if(NOT DEFINED FCITX5_ANDROID_SDK_DIR)
            set(FCITX5_ANDROID_SDK_DIR "$ENV{HOME}/.local/share/fcitx5-llm-completion/fcitx5-android-sdk")
        endif()
        if(NOT EXISTS "${FCITX5_ANDROID_SDK_DIR}/lib/${ANDROID_ABI}/libFcitx5Core.so")
            message(FATAL_ERROR "Missing Fcitx5 Android SDK at ${FCITX5_ANDROID_SDK_DIR}/lib/${ANDROID_ABI}")
        endif()
        add_library(Fcitx5::Utils SHARED IMPORTED)
        set_target_properties(Fcitx5::Utils PROPERTIES
            IMPORTED_LOCATION "${FCITX5_ANDROID_SDK_DIR}/lib/${ANDROID_ABI}/libFcitx5Utils.so"
            INTERFACE_INCLUDE_DIRECTORIES "${FCITX5_ANDROID_SDK_DIR}/include/Fcitx5/Utils"
        )
        add_library(Fcitx5::Core SHARED IMPORTED)
        set_target_properties(Fcitx5::Core PROPERTIES
            IMPORTED_LOCATION "${FCITX5_ANDROID_SDK_DIR}/lib/${ANDROID_ABI}/libFcitx5Core.so"
            INTERFACE_INCLUDE_DIRECTORIES "${FCITX5_ANDROID_SDK_DIR}/include/Fcitx5/Core;${FCITX5_ANDROID_SDK_DIR}/include/Fcitx5/Config"
            INTERFACE_LINK_LIBRARIES Fcitx5::Utils
        )
    else()
        find_package(Fcitx5Core REQUIRED)
        find_package(Fcitx5Utils REQUIRED)
    endif()

    add_library(llmcompletion SHARED src/addon/llmcompletion.cpp)
    set_target_properties(llmcompletion PROPERTIES
        OUTPUT_NAME "llmcompletion"
        PREFIX "lib"
    )
    target_compile_options(llmcompletion PRIVATE -Wall -Wextra -O3)
    target_link_libraries(llmcompletion PRIVATE
        fcitx5_llm_core
        Fcitx5::Core
        Fcitx5::Utils
    )
    if(NOT ANDROID AND NOT "${_OV_LIB_RPATH}" STREQUAL "")
        set_target_properties(llmcompletion PROPERTIES
            BUILD_RPATH "${_OV_LIB_RPATH}"
            INSTALL_RPATH "${_OV_LIB_RPATH}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()

if(FCITX5_LLM_BUILD_TESTS AND NOT ANDROID)
    enable_testing()
    add_executable(fcitx5_llm_tests tests/core_test.cpp)
    target_compile_options(fcitx5_llm_tests PRIVATE -Wall -Wextra)
    target_link_libraries(fcitx5_llm_tests PRIVATE fcitx5_llm_core)
    if(NOT "${_OV_LIB_RPATH}" STREQUAL "")
        set_target_properties(fcitx5_llm_tests PROPERTIES
            BUILD_RPATH "${_OV_LIB_RPATH}"
            INSTALL_RPATH "${_OV_LIB_RPATH}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
    add_test(NAME fcitx5_llm_tests COMMAND fcitx5_llm_tests)
endif()
