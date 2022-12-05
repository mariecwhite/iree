#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Downloads SHARK raw results and retroactively updates the summaries. This
# can be useful when updating baselines or adding new metrics.

set -xeuo pipefail

# Download baselines.
shark_baseline_dir="/tmp/shark_baselines"
mkdir -p "${shark_baseline_dir}"
gsutil cp -r "gs://shark-benchmark-artifacts/baselines/**" "${shark_baseline_dir}"

# Download benchmark results.
tmp_dir="/tmp/shark_tmp"
mkdir -p "${tmp_dir}"

declare -a gcs_dirs=($(gsutil ls gs://shark-benchmark-artifacts | grep "^gs://shark-benchmark-artifacts/.*.sha_.*.timestamp_.*/$"))

#for gcs_dir in "${gcs_dirs[@]}"; do
#  gsutil cp -r "${gcs_dir}" "${tmp_dir}"
#done

python3 -m venv iree.venv
source iree.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade -r "build_tools/benchmarks/reporting/requirements.txt"

# Update benchmark results.
declare -a local_dirs=($(ls "${tmp_dir}"))
for dir_name in "${local_dirs[@]}"; do
  if [[ "${dir_name}" =~ .*manual$ ]]; then
    echo "Skipping ${dir_name}"
    continue
  fi

  dir="${tmp_dir}/${dir_name}"
  echo "${dir}"

  # Backup previous summary.html file.
  mv "${dir}/summary.html" "${dir}/summary.html.bak.$(date +'%s')" || true

  cpu_iree_path=$(ls -d ${dir}/* | grep cpu_iree)
  cpu_shark_path=$(ls -d ${dir}/* | grep cpu_shark)
  cpu_baseline="${shark_baseline_dir}/cpu_baseline.csv"
  cuda_iree_path=$(ls -d ${dir}/* | grep cuda_iree)
  cuda_shark_path=$(ls -d ${dir}/* | grep cuda_shark)
  cuda_baseline="${shark_baseline_dir}/cuda_baseline.csv"

  iree_version=$(echo "${cpu_iree_path}" | sed "s/.*cpu_iree_\(.*\).csv$/\1/")
  shark_version=$(echo "${cpu_shark_path}" | sed "s/.*cpu_shark_\(.*\).csv$/\1/")
  timestamp=$(echo "${dir_name}" | sed "s/.*timestamp_\(.*\)$/\1/")

  python3 build_tools/benchmarks/reporting/parse_shark_benchmarks.py \
    --timestamp="${timestamp}" \
    --shark_version="${shark_version}" \
    --iree_version="${iree_version}" \
    --cpu_shark_csv="${cpu_shark_path}" \
    --cpu_iree_csv="${cpu_iree_path}" \
    --cpu_baseline="${cpu_baseline}" \
    --gpu_shark_csv="${cuda_shark_path}" \
    --gpu_iree_csv="${cuda_iree_path}" \
    --gpu_baseline="${cuda_baseline}" \
    --output_dir="${dir}"

  gsutil cp "${dir}/**" "gs://shark-benchmark-artifacts/${dir_name}/"
done
