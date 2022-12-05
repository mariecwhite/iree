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

base_dir="/tmp/shark_summaries"
rm -rf "${base_dir}"
mkdir "${base_dir}"

summaries_dir="${base_dir}/summaries"
trends_dir="${base_dir}/trends"
mkdir -p "${summaries_dir}"
mkdir -p "${trends_dir}"

# Download benchmark results.
declare -a gcs_dirs=($(gsutil ls gs://shark-benchmark-artifacts | grep "^gs://shark-benchmark-artifacts/.*.sha_.*.timestamp_.*/$"))

for gcs_dir in "${gcs_dirs[@]}"; do
  if [[ "${gcs_dir}" =~ .*manual/$ ]]; then
    echo "Skipping ${gcs_dir}"
    continue
  fi

  version=$(echo "${gcs_dir}" | sed "s/gs:\/\/shark-benchmark-artifacts\/\(.*\)\/$/\1/")
  gsutil cp -r "${gcs_dir}summary.csv" "${summaries_dir}/${version}.csv"
  gsutil cp -r "${gcs_dir}trends.csv" "${trends_dir}/${version}.csv"
done

# Upload to new source directory
gcs_upload_dir="gs://shark-benchmark-artifacts/summaries/$(date +'%s')"
gsutil cp -r "${summaries_dir}/**" "${gcs_upload_dir}/summaries/"
gsutil cp -r "${trends_dir}/**" "${gcs_upload_dir}/trends/"
