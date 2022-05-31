# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/bin/bash

set -euo pipefail

# Install Bazel. From https://www.tensorflow.org/install/source
npm install -g @bazel/bazelisk

# Create root dir.
ROOT_DIR=/tmp/mobilebert_benchmarks
mkdir ${ROOT_DIR}
mkdir ${ROOT_DIR}/models
mkdir ${ROOT_DIR}/models/tflite
mkdir ${ROOT_DIR}/models/iree
mkdir ${ROOT_DIR}/models/iree/dylib
mkdir ${ROOT_DIR}/test_data
mkdir ${ROOT_DIR}/output

# Assumes all models are pre-built.
OCR_MODEL_DIR=/usr/local/google/home/mariewhite/ocr/models
cp ${OCR_MODEL_DIR}/mobilenet_quant_v1_224.tflite ${ROOT_DIR}/models/tflite/
cp ${OCR_MODEL_DIR}/x86_64/mobilenet_quant_v1_224.vmfb ${ROOT_DIR}/models/iree/dylib/
cp ${OCR_MODEL_DIR}/rpn_text_detector_mobile_space_to_depth_float_v2.tflite ${ROOT_DIR}/models/tflite/
cp ${OCR_MODEL_DIR}/x86_64/rpn_text_detector_mobile_space_to_depth_float_v2.vmfb ${ROOT_DIR}/models/iree/dylib/
cp ${OCR_MODEL_DIR}/rpn_text_detector_mobile_space_to_depth_quantized_mbv2_v1.tflite ${ROOT_DIR}/models/tflite/
cp ${OCR_MODEL_DIR}/x86_64/rpn_text_detector_mobile_space_to_depth_quantized_mbv2_v1.vmfb ${ROOT_DIR}/models/iree/dylib/
cp ${OCR_MODEL_DIR}/rpn_text_detector_mobile_space_to_depth_quantized_v2.tflite ${ROOT_DIR}/models/tflite/
cp ${OCR_MODEL_DIR}/x86_64/rpn_text_detector_mobile_space_to_depth_quantized_v2.vmfb ${ROOT_DIR}/models/iree/dylib/
cp ${OCR_MODEL_DIR}/tflite_lstm_recognizer_latin_0.3.conv_model.tflite ${ROOT_DIR}/models/tflite/
cp ${OCR_MODEL_DIR}/x86_64/tflite_lstm_recognizer_latin_0.3.conv_model.vmfb ${ROOT_DIR}/models/iree/dylib/
cp ${OCR_MODEL_DIR}/tflite_screen_recognizer_latin.conv_model.tflite ${ROOT_DIR}/models/tflite/
cp ${OCR_MODEL_DIR}/x86_64/tflite_screen_recognizer_latin.conv_model.vmfb ${ROOT_DIR}/models/iree/dylib/

SOURCE_DIR=/usr/local/google/home/mariewhite/github
cp ${SOURCE_DIR}/iree-build/iree/tools/iree-benchmark-module ${ROOT_DIR}/

# Build TFLite benchmark.
cd ${SOURCE_DIR}/tensorflow

bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model
cp ${SOURCE_DIR}/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model ${ROOT_DIR}/

# Run benchmark.
cd "${SOURCE_DIR}/iree"
python3.9 build_tools/benchmarks/comparisons/run_benchmarks.py \
  --device_name=desktop --base_dir=${ROOT_DIR} \
  --output_dir=${ROOT_DIR}/output --mode=desktop

cat "${ROOT_DIR}/output/results.csv"
