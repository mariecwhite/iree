# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Run commands below on the workstation that the phone is attached to.
# Prerequisites:
#   Manual installations of the Android NDK and ADB are needed. See https://google.github.io/iree/building-from-source/android/#install-android-ndk-and-adb for instructions.
#   Manual installations of the Termux App and python are needed on the Android device. See README.md for instructions.

#!/bin/bash

set -euo pipefail

GPU_TYPE="mali"
#GPU_TYPE="andreno"

# Create root dir.
ROOT_DIR=/tmp/ocr_benchmarks

rm -rf ${ROOT_DIR}
mkdir ${ROOT_DIR}
mkdir ${ROOT_DIR}/models
mkdir ${ROOT_DIR}/models/tflite
mkdir ${ROOT_DIR}/models/iree
mkdir ${ROOT_DIR}/models/iree/dylib
mkdir ${ROOT_DIR}/models/iree/vulkan
mkdir ${ROOT_DIR}/setup
mkdir ${ROOT_DIR}/test_data
mkdir ${ROOT_DIR}/output

# Assumes all models are pre-built.
OCR_MODEL_DIR=/usr/local/google/home/mariewhite/ocr/models
cp ${OCR_MODEL_DIR}/gocr_tflite_recognizer_latin_float.tflite ${ROOT_DIR}/models/tflite/
cp ${OCR_MODEL_DIR}/aarch64/gocr_tflite_recognizer_latin_float.vmfb ${ROOT_DIR}/models/iree/dylib/
cp ${OCR_MODEL_DIR}/aarch64/gocr_tflite_recognizer_latin_float_mmt4d.vmfb ${ROOT_DIR}/models/iree/dylib/
cp ${OCR_MODEL_DIR}/vulkan_${GPU_TYPE}/gocr_tflite_recognizer_latin_float.vmfb ${ROOT_DIR}/models/iree/vulkan/
#cp ${OCR_MODEL_DIR}/mobilenet_quant_v1_224.tflite ${ROOT_DIR}/models/tflite/
#cp ${OCR_MODEL_DIR}/aarch64/mobilenet_quant_v1_224.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/aarch64/mobilenet_quant_v1_224_mmt4d.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/vulkan_${GPU_TYPE}/mobilenet_quant_v1_224.vmfb ${ROOT_DIR}/models/iree/vulkan/
#cp ${OCR_MODEL_DIR}/rpn_text_detector_mobile_space_to_depth_float_v2.tflite ${ROOT_DIR}/models/tflite/
#cp ${OCR_MODEL_DIR}/aarch64/rpn_text_detector_mobile_space_to_depth_float_v2.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/aarch64/rpn_text_detector_mobile_space_to_depth_float_v2_mmt4d.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/vulkan_${GPU_TYPE}/rpn_text_detector_mobile_space_to_depth_float_v2.vmfb ${ROOT_DIR}/models/iree/vulkan/
#cp ${OCR_MODEL_DIR}/rpn_text_detector_mobile_space_to_depth_quantized_mbv2_v1.tflite ${ROOT_DIR}/models/tflite/
#cp ${OCR_MODEL_DIR}/aarch64/rpn_text_detector_mobile_space_to_depth_quantized_mbv2_v1.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/aarch64/rpn_text_detector_mobile_space_to_depth_quantized_mbv2_v1_mmt4d.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/vulkan_${GPU_TYPE}/rpn_text_detector_mobile_space_to_depth_quantized_mbv2_v1.vmfb ${ROOT_DIR}/models/iree/vulkan/
#cp ${OCR_MODEL_DIR}/rpn_text_detector_mobile_space_to_depth_quantized_v2.tflite ${ROOT_DIR}/models/tflite/
#cp ${OCR_MODEL_DIR}/aarch64/rpn_text_detector_mobile_space_to_depth_quantized_v2.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/aarch64/rpn_text_detector_mobile_space_to_depth_quantized_v2_mmt4d.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/vulkan_${GPU_TYPE}/rpn_text_detector_mobile_space_to_depth_quantized_v2.vmfb ${ROOT_DIR}/models/iree/vulkan/
#cp ${OCR_MODEL_DIR}/tflite_lstm_recognizer_latin_0.3.conv_model.tflite ${ROOT_DIR}/models/tflite/
#cp ${OCR_MODEL_DIR}/aarch64/tflite_lstm_recognizer_latin_0.3.conv_model.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/aarch64/tflite_lstm_recognizer_latin_0.3_mmt4d.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/vulkan_${GPU_TYPE}/tflite_lstm_recognizer_latin_0.3.conv_model.vmfb ${ROOT_DIR}/models/iree/vulkan/
#cp ${OCR_MODEL_DIR}/tflite_screen_recognizer_latin.conv_model.tflite ${ROOT_DIR}/models/tflite/
#cp ${OCR_MODEL_DIR}/aarch64/tflite_screen_recognizer_latin.conv_model.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/aarch64/tflite_screen_recognizer_latin.conv_model_mmt4d.vmfb ${ROOT_DIR}/models/iree/dylib/
#cp ${OCR_MODEL_DIR}/vulkan_${GPU_TYPE}/tflite_screen_recognizer_latin.conv_model.vmfb ${ROOT_DIR}/models/iree/vulkan/

# Build IREE source.
SOURCE_DIR=/usr/local/google/home/mariewhite/github

cp ${SOURCE_DIR}/iree/build_tools/benchmarks/set_adreno_gpu_scaling_policy.sh ${ROOT_DIR}/setup/
cp ${SOURCE_DIR}/iree/build_tools/benchmarks/set_android_scaling_governor.sh ${ROOT_DIR}/setup/
cp ${SOURCE_DIR}/iree/build_tools/benchmarks/set_pixel6_gpu_scaling_policy.sh ${ROOT_DIR}/setup/

# Cross-compile IREE benchmark binary.
cd "${SOURCE_DIR}/iree"
cmake -GNinja -B ../iree-build/ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  .
cmake --build ../iree-build/ --target install

rm -rf ${SOURCE_DIR}/iree-build-android

export ANDROID_NDK=/usr/local/google/home/mariewhite/Android/Sdk/ndk/20.1.5948944
cmake -GNinja -B ../iree-build-android/ \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT="${PWD}/../iree-build/install" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM="android-29" \
  -DIREE_BUILD_COMPILER=OFF \
  .
cmake --build ../iree-build-android/
cp "${SOURCE_DIR}/iree-build-android/tools/iree-benchmark-module" "${ROOT_DIR}/"

# Cross-compile TFLite benchmark binary.
sudo apt-get install libgles2-mesa-dev

export CC=clang
export CXX=clang++

cd ${SOURCE_DIR}/tensorflow
# Select defaults. Answer Yes to configuring ./WORKSPACE for Android builds.
# Use Version 21 for Android NDK, 29 for Android SDK.
python configure.py
bazel build -c opt --config=android_arm64 \
  --copt="-Wno-error=implicit-function-declaration" \
  tensorflow/lite/tools/benchmark:benchmark_model

cp "${SOURCE_DIR}/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model" "${ROOT_DIR}/"

# Push benchmarking artifacts to device.
adb shell rm -r /data/local/tmp/mobilebert_benchmarks
adb push "${ROOT_DIR}" /data/local/tmp

DEVICE_ROOT_DIR=/data/local/tmp/mobilebert_benchmarks
adb shell chmod +x "${DEVICE_ROOT_DIR}/benchmark_model"
adb shell chmod +x "${DEVICE_ROOT_DIR}/iree-benchmark-module"

# Setup device.
adb shell "su root sh ${DEVICE_ROOT_DIR}/setup/set_android_scaling_governor.sh performance"

if [[ "${GPU_TYPE}" = "mali" ]]; then
  adb shell "su root sh ${DEVICE_ROOT_DIR}/setup/set_pixel6_gpu_scaling_policy.sh performance"
else
  adb shell "su root sh ${DEVICE_ROOT_DIR}/setup/set_adreno_gpu_scaling_policy.sh performance"
fi

# Run benchmark.
adb push "${SOURCE_DIR}/build_tools/benchmarks/comparisons" /data/local/tmp/
adb shell "su root /data/data/com.termux/files/usr/bin/python /data/local/tmp/comparisons/run_benchmarks.py --device_name=Pixel6  --mode=mobile --base_dir=${DEVICE_ROOT_DIR} --output_dir=${DEVICE_ROOT_DIR}/output"
adb shell cat "${DEVICE_ROOT_DIR}/output/result.csv"

