# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
""" Runs benchmarks and saves results to a .csv file

Expects a directory structure of:
<root-benchmark-dir>/
  └── ./benchmark_model (TFLite benchmark binary)
      ./iree-benchmark-module (IREE benchmark binary)
  ├── setup/
        ├── set_adreno_gpu_scaling_policy.sh
        ├── set_android_scaling_governor.sh
        └── set_pixel6_gpu_scaling_policy.sh
  ├── test_data/
  └── models/
        ├── tflite/*.tflite
        └── iree/
              └── <driver>/*.vmfb e.g. dylib, vulkan, cuda.

"""

import argparse
import os

from common.benchmark_runner import *
from common.utils import *
from gocr_tflite_recognizer_latin_float import *
from mobilebert_fp32_commands import *
from ocr_mobilenet_quant_v1_224 import *
from ocr_rpn_text_detector_mobile_space_to_depth_float_v2 import *
from ocr_rpn_text_detector_mobile_space_to_depth_quantized_mbv2_v1 import *
from ocr_rpn_text_detector_mobile_space_to_depth_quantized_v2 import *
from ocr_tflite_lstm_recognizer_latin_03_conv_model import *
from ocr_tflite_screen_recognizer_latin_conv_model import *

def benchmark_desktop_cpu(device_name: str,
                          command_factories: list[BenchmarkCommandFactory],
                          results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("desktop", "cpu"))

  for num_threads in [1, 2, 4, 8]:
    for benchmark in benchmarks:
      results_array = [
          device_name, benchmark.model_name, benchmark.runtime,
          benchmark.driver, num_threads
      ]
      benchmark.num_threads = num_threads
      results_array.extend(run_command(benchmark))
      write_benchmark_result(results_array, results_path)


def benchmark_desktop_gpu(device_name: str,
                          command_factories: list[BenchmarkCommandFactory],
                          results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("desktop", "gpu"))
  for benchmark in benchmarks:
    results_array = [
        device_name, benchmark.model_name, benchmark.runtime, benchmark.driver,
        benchmark.num_threads
    ]
    results_array.extend(run_command(benchmark))
    write_benchmark_result(results_array, results_path)


def benchmark_mobile_cpu(device_name: str,
                         command_factories: list[BenchmarkCommandFactory],
                         results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("mobile", "cpu"))

  for _, tuple in enumerate([("80", 1), ("C0", 2), ("F0", 4), ("0F", 4),
                             ("FF", 8)]):
    taskset = tuple[0]
    num_threads = tuple[1]
    for benchmark in benchmarks:
      results_array = [
          device_name, benchmark.model_name, benchmark.runtime,
          benchmark.driver, taskset, num_threads
      ]
      benchmark.taskset = taskset
      benchmark.num_threads = num_threads
      results_array.extend(run_command(benchmark))
      write_benchmark_result(results_array, results_path)


def benchmark_mobile_gpu(device_name: str,
                         command_factories: list[BenchmarkCommandFactory],
                         results_path: str):
  benchmarks = []
  for factory in command_factories:
    benchmarks.extend(factory.generate_benchmark_commands("mobile", "gpu"))

  taskset = "80"
  num_threads = 1
  for benchmark in benchmarks:
    results_array = [
        device_name, benchmark.model_name, benchmark.runtime, benchmark.driver,
        taskset, num_threads
    ]
    benchmark.taskset = taskset
    benchmark.num_threads = num_threads
    results_array.extend(run_command(benchmark))
    write_benchmark_result(results_array, results_path)


def main(args):
  # Create factories for all models to be benchmarked.
  command_factory = []
  command_factory.append(GocrTfliteRecognizerLatinFloatFactory(args.base_dir))
  #command_factory.append(MobilebertFP32CommandFactory(args.base_dir))
  #command_factory.append(MobilenetQuantV1224Factory(args.base_dir))
  #command_factory.append(RpnTextDetectorMobileSpaceToDepthFloatV2Factory(args.base_dir))
  #command_factory.append(RpnTextDetectorMobileSpaceToDepthQuantizedMbv2V1Factory(args.base_dir))
  #command_factory.append(RpnTextDetectorMobileSpaceToDepthQuantizedV2Factory(args.base_dir))
  #command_factory.append(TfliteLstmRecognizerLatin03ConvModelFactory(args.base_dir))
  #command_factory.append(TfliteScreenRecognizerLatinConvModelFactory(args.base_dir))

  if args.mode == "desktop":
    results_path = os.path.join(args.output_dir, "results.csv")
    with open(results_path, "w") as f:
      f.write(
          "device,model,runtime,driver/delegate,threads,latency (ms),vmhwm (KB),vmrss (KB),rssfile (KB)\n"
      )

    if not args.disable_cpu:
      benchmark_desktop_cpu(args.device_name, command_factory, results_path)
    if not args.disable_gpu:
      benchmark_desktop_gpu(args.device_name, command_factory, results_path)
  else:
    assert (args.mode == "mobile")
    results_path = os.path.join(args.output_dir, "results.csv")
    with open(results_path, "w") as f:
      f.write(
          "device,model,runtime,driver/delegate,taskset,threads,latency (ms),vmhwm (KB),vmrss (KB),rssfile (KB)\n"
      )
    if not args.disable_cpu:
      benchmark_mobile_cpu(args.device_name, command_factory, results_path)
    if not args.disable_gpu:
      benchmark_mobile_gpu(args.device_name, command_factory, results_path)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--device_name",
      type=str,
      default=None,
      help="The name of the device the benchmark is running on e.g. Pixel 6")
  parser.add_argument(
      "--base_dir",
      type=str,
      default=None,
      help="The directory where all benchmarking artifacts are located.")
  parser.add_argument("--output_dir",
                      type=str,
                      default=None,
                      help="The directory to save output artifacts into.")
  parser.add_argument(
      "--mode",
      type=str,
      choices=("desktop", "mobile"),
      default="desktop",
      help="The benchmarking mode to use. If mode is `mobile`, uses tasksets.")
  parser.add_argument("--disable_cpu",
                      action="store_true",
                      help="Disables running benchmarks on CPU.")
  parser.add_argument("--disable_gpu",
                      action="store_true",
                      help="Disables running benchmarks on GPU.")
  return parser.parse_args()


if __name__ == '__main__':
  main(parse_args())
