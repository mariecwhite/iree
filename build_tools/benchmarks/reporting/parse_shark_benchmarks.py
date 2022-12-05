# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Script to generate a HTML summary of SHARK Tank benchmarks.

Example usage:

python parse_shark_benchmarks.py \
  --shark_version=20220924.162 \
  --iree_version=20220924.276 \
  --timestamp=1666951260 \
  --cpu_shark_csv=icelake_shark_bench_results.csv \
  --cpu_iree_csv=icelake_iree_bench_results.csv \
  --cpu_baseline=cpu_baseline.csv \
  --gpu_shark_csv=a100_shark_bench_results.csv \
  --gpu_iree_csv=a100_iree_bench_results.csv \
  --gpu_baseline=a100_baseline.csv \
  --output_dir=/tmp/shark_summary

"""

import argparse
import os
import pandas as pd
import pathlib
import scipy
import sys

from datetime import date

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "python"))
from reporting.common import html_utils

# Column headers.
_MODEL = "model"
_DIALECT = "dialect"
_DATA_TYPE = "data_type"
_SHAPE_TYPE = "shape_type"
_BASELINE = "baseline"
_DEVICE = "device"
_BASELINE_LATENCY = "baseline latency (ms)"
_IREE_LATENCY = "IREE latency (ms)"
_SHARK_LATENCY = "SHARK latency (ms)"
_IREE_VS_BASELINE = "IREE vs baseline"
_IREE_VS_BASELINE_RAW = "IREE vs baseline (x faster)"
_SHARK_VS_BASELINE = "SHARK vs baseline"
_IREE_VS_SHARK = "IREE vs SHARK"
_GEOMEAN = "geomean"
_METRIC_TYPE = "metric_type"
_METRIC_UNITS = "metric_units"
_METRIC_VALUE = "metric_value"
_TIMESTAMP = "timestamp"

_PERF_COLUMNS = [_IREE_VS_BASELINE, _SHARK_VS_BASELINE, _IREE_VS_SHARK]
_LATENCY_COLUMNS = [_BASELINE_LATENCY, _IREE_LATENCY, _SHARK_LATENCY]


def _create_summary_df():
  df = pd.DataFrame(columns=[
      _MODEL,
      _BASELINE,
      _DATA_TYPE,
      _SHAPE_TYPE,
      _DIALECT,
      _DEVICE,
      _BASELINE_LATENCY,
      _IREE_LATENCY,
      _SHARK_LATENCY,
      _IREE_VS_BASELINE,
      _SHARK_VS_BASELINE,
      _IREE_VS_SHARK,
      _IREE_VS_BASELINE_RAW,
      _TIMESTAMP,
  ])
  df = df.astype({
      _IREE_VS_BASELINE_RAW: 'float',
      _BASELINE_LATENCY: 'float',
      _IREE_LATENCY: 'float',
      _SHARK_LATENCY: 'float',
  })
  return df


def _create_trends_df():
  df = pd.DataFrame(columns=[
      _DEVICE, _SHAPE_TYPE, _BASELINE, _METRIC_TYPE, _METRIC_UNITS,
      _METRIC_VALUE, _TIMESTAMP
  ])
  df = df.astype({_METRIC_VALUE: 'float'})
  return df


def _generate_table(timestamp, df_iree, df_shark, df_baseline, shape_type, title):
  """Generates a table comparing latencies between IREE, SHARK and a baseline."""
  summary = _create_summary_df()

  models = df_iree.model.unique()
  for model in models:
    iree_results_per_model = df_iree.loc[df_iree.model == model]
    dialects = iree_results_per_model.dialect.unique()
    for dialect in dialects:
      iree_results_per_dialect = iree_results_per_model.loc[
          iree_results_per_model.dialect == dialect]
      data_types = iree_results_per_dialect.data_type.unique()
      for data_type in data_types:
        iree_results_per_datatype = iree_results_per_dialect.loc[
            iree_results_per_dialect.data_type == data_type]
        device_types = iree_results_per_datatype.device.unique()
        for device in device_types:
          iree_results = iree_results_per_datatype.loc[
              iree_results_per_datatype.device == device]
          if len(iree_results) != 3:
            print(f"Warning! Expected number of results to be 3. Got"
                  f" {len(iree_results)}")
            print(iree_results)
            continue

          baseline_results = df_baseline.loc[(df_baseline.model == model) &
                                             (df_baseline.dialect == dialect) &
                                             (df_baseline.data_type
                                              == data_type) &
                                             (df_baseline.device == device)]

          if baseline_results.empty:
            # We use snapshots of latencies for baseline. If it is a new
            # benchmark that is not included in the snapshot yet, emit a
            # warning.
            print(
                f"Warning: No baseline results found for {model}, {dialect},"
                f" {data_type}, {device}. Using IREE version as baseline. Please"
                f" update baseline csv.")
            engine = iree_results.engine.iloc[0]
            baseline_latency = iree_results.loc[iree_results.engine == engine]
            baseline_latency = baseline_latency.iloc[0]["ms/iter"]
          else:
            engine = baseline_results.engine.iloc[0]
            baseline_latency = baseline_results.loc[baseline_results.engine ==
                                                    engine]
            baseline_latency = baseline_latency.iloc[0]["ms/iter"]

          iree_latency = iree_results.loc[iree_results.engine == "shark_iree_c"]
          iree_latency = iree_latency.iloc[0]["ms/iter"]
          iree_vs_baseline = html_utils.format_latency_comparison(
              iree_latency, baseline_latency)
          iree_vs_baseline_raw = baseline_latency / iree_latency


          if df_shark is not None:
            shark_results = df_shark.loc[(df_shark.model == model) &
                                         (df_shark.dialect == dialect) &
                                         (df_shark.data_type == data_type) &
                                         (df_shark.device == device)]
            if shark_results.empty:
              print(
                  f"Warning: No SHARK results for {model}, {dialect}, {data_type}, {device}."
              )
              continue

            shark_latency = shark_results.loc[shark_results.engine ==
                                              "shark_iree_c"]
            shark_latency = shark_latency.iloc[0]["ms/iter"]
            shark_vs_baseline = html_utils.format_latency_comparison(
                shark_latency, baseline_latency)
            iree_vs_shark = html_utils.format_latency_comparison(
                iree_latency, shark_latency)
          else:
            # If there are no SHARK benchmarks available, use default values.
            # These columns will be hidden later.
            shark_latency = 0
            shark_vs_baseline = "<missing_comparison>"
            iree_vs_shark = "<missing_comparison>"

          summary.loc[len(summary)] = [
              model, engine, data_type, shape_type, dialect, device,
              round(baseline_latency, 1),
              round(iree_latency, 1),
              round(shark_latency, 1), iree_vs_baseline, shark_vs_baseline,
              iree_vs_shark, iree_vs_baseline_raw, timestamp
          ]

  st = summary.style.set_table_styles(html_utils.get_table_css())
  st = st.format(precision=1)
  st = st.hide(axis="index")
  st = st.hide_columns(subset=[_IREE_VS_BASELINE_RAW, _TIMESTAMP])
  if df_shark is None:
    st = st.hide_columns(
        subset=[_SHARK_LATENCY, _SHARK_VS_BASELINE, _IREE_VS_SHARK])
  st = st.set_caption(title)
  st = st.applymap(html_utils.style_performance, subset=_PERF_COLUMNS)
  st = st.set_properties(subset=[_MODEL],
                         **{
                             "width": "300px",
                             "text-align": "left",
                         })
  st = st.set_properties(subset=[_BASELINE],
                         **{
                             "width": "140",
                             "text-align": "center",
                         })
  st = st.set_properties(subset=[_DIALECT, _DATA_TYPE, _DEVICE],
                         **{
                             "width": "100",
                             "text-align": "center",
                         })
  st = st.set_properties(subset=_LATENCY_COLUMNS,
                         **{
                             "width": "100",
                             "text-align": "right",
                         })
  st = st.set_properties(subset=_PERF_COLUMNS,
                         **{
                             "width": "150px",
                             "text-align": "right",
                             "color": "#ffffff"
                         })
  html = st.to_html() + "<br/>"

  return summary, html


def generate_table(timestamp,
                   iree_csv,
                   baseline_csv,
                   shark_csv=None,
                   shape_type="static",
                   device="cpu",
                   title="Benchmarks"):
  """Generates a table comparing latencies between IREE, SHARK and a baseline.

  Args:
    timestamp: Timestamp results were captured.
    iree_csv: Path to the csv file containing IREE latencies.
    baseline_csv: Path to the csv file containing baseline latencies.
    shark_csv: Path to the csv file containing SHARK-Runtime latencies. This is optional.
    shape_type: Currently either `static` or `dynamic`.
    device: Device used to run the benchmarks.
    title: The title of the generated table.

  Returns:
    A tuple containing a summary dataframe and an HTML string representing the
    summarized report.
  """
  shark_df = None
  if shark_csv is not None:
    shark_df = pd.read_csv(shark_csv)
    shark_df = shark_df.loc[(shark_df.shape_type == shape_type) &
                            (shark_df.device == device)]

  iree_df = pd.read_csv(iree_csv)
  iree_df = iree_df.loc[(iree_df.shape_type == shape_type) &
                        (iree_df.device == device)]

  baseline_df = pd.read_csv(baseline_csv)
  baseline_df = baseline_df.loc[(baseline_df.shape_type == shape_type) &
                                (baseline_df.device == device)]

  return _generate_table(timestamp, iree_df, shark_df, baseline_df, shape_type, title)


def _calculate_geomean(timestamp, summary_df, device, baseline, shape_type, trends_df):
  df = summary_df.loc[(summary_df.device == device) &
                      (summary_df.baseline == baseline) &
                      (summary_df.shape_type == shape_type)]

  if df.empty:
    print(f"Could not calculate geomean for {device}, {baseline}, {shape_type}. Dataframe empty.")
    return trends_df

  geomean = scipy.stats.mstats.gmean(df.loc[:, _IREE_VS_BASELINE_RAW])
  trends_df.loc[len(trends_df)] = [
      device, shape_type, baseline, "geomean", "x faster", geomean, timestamp
  ]
  return trends_df


def _calculate_static_vs_dynamic(timestamp, summary_df, device, trends_df):
  dynamic_df = summary_df.loc[(summary_df.device == device) &
                              (summary_df.baseline == "torch") &
                              (summary_df.shape_type == "dynamic")]
  static_df = summary_df.loc[(summary_df.device == device) &
                             (summary_df.baseline == "torch") &
                             (summary_df.shape_type == "static")]

  # Only include models that exist in both static and dynamic modes.
  dynamic_df = dynamic_df[dynamic_df[_MODEL].isin(static_df[_MODEL])]
  static_df = static_df[static_df[_MODEL].isin(static_df[_MODEL])]

  if dynamic_df.empty or static_df.empty:
    print(f"Could not calculate mean for {device}. Dataframe empty.")
    return trends_df

  iree_dynamic_mean_latency = dynamic_df[_IREE_LATENCY].mean()
  trends_df.loc[len(trends_df)] = [
      device, "dynamic", "iree", "mean", "ms", iree_dynamic_mean_latency, timestamp
  ]

  torch_dynamic_mean_latency = dynamic_df[_BASELINE_LATENCY].mean()
  trends_df.loc[len(trends_df)] = [
      device, "dynamic", "torch", "mean", "ms", torch_dynamic_mean_latency, timestamp
  ]

  iree_static_mean_latency = static_df[_IREE_LATENCY].mean()
  trends_df.loc[len(trends_df)] = [
      device, "static", "iree", "mean", "ms", iree_static_mean_latency, timestamp
  ]

  torch_static_mean_latency = static_df[_BASELINE_LATENCY].mean()
  trends_df.loc[len(trends_df)] = [
      device, "static", "torch", "mean", "ms", torch_static_mean_latency, timestamp
  ]

  return trends_df


def generate_trends(timestamp, summary_df):
  trends_df = _create_trends_df()

  # Static shapes.
  # Calculate geomean of IREE vs PyTorch on CPU.
  trends_df = _calculate_geomean(timestamp, summary_df, "cpu", "torch", "static",
                                 trends_df)
  # Calculate geomean of IREE vs PyTorch on Cuda.
  trends_df = _calculate_geomean(timestamp, summary_df, "cuda", "torch", "static",
                                 trends_df)
  # Calculate geomean of IREE vs TF/XLA on CPU.
  trends_df = _calculate_geomean(timestamp, summary_df, "cpu", "tf", "static", trends_df)
  # Calculate geomean of IREE vs TF/XLA on Cuda.
  trends_df = _calculate_geomean(timestamp, summary_df, "cuda", "tf", "static", trends_df)

  # Dynamic shapes. Dynamic shapes only work on Torch models at the moment.
  # Calculate geomean of IREE vs PyTorch on CPU.
  trends_df = _calculate_geomean(timestamp, summary_df, "cpu", "torch", "dynamic",
                                 trends_df)
  # Calculate geomean of IREE vs PyTorch on Cuda.
  trends_df = _calculate_geomean(timestamp, summary_df, "cuda", "torch", "dynamic",
                                 trends_df)

  # Dynamic vs static.
  trends_df = _calculate_static_vs_dynamic(timestamp, summary_df, "cpu", trends_df)
  trends_df = _calculate_static_vs_dynamic(timestamp, summary_df, "cuda", trends_df)

  st = trends_df.style.set_table_styles(html_utils.get_table_css())
  st = st.format(precision=1)
  st = st.hide(axis="index")
  st = st.hide_columns(subset=[_TIMESTAMP])
  st = st.set_caption("Overall Metrics")
  html = st.to_html() + "<br/>"

  return trends_df, html


def main(args):
  """Summarizes benchmark results generated by the SHARK Tank."""
  verison_html = (f"<i>IREE version: {args.iree_version}</i><br/>"
                  f"<i>SHARK version: {args.shark_version}</i><br/>"
                  f"<i>last updated: {date.today().isoformat()}</i><br/><br/>")

  main_html = html_utils.generate_header_and_legend(verison_html)
  summary_html = ""
  df = _create_summary_df()

  # Generate Server CPU Static.
  if args.cpu_iree_csv is not None:
    summary_df, html = generate_table(
        args.timestamp,
        args.cpu_iree_csv,
        args.cpu_baseline_csv,
        shark_csv=args.cpu_shark_csv,
        shape_type="static",
        device="cpu",
        title="Server Intel Ice Lake CPU (Static Shapes)")
    df = df.append(summary_df, ignore_index=True)
    summary_html += html

  # Generate Server GPU Static.
  if args.gpu_iree_csv is not None:
    summary_df, html = generate_table(
        args.timestamp,
        args.gpu_iree_csv,
        args.gpu_baseline_csv,
        shark_csv=args.gpu_shark_csv,
        shape_type="static",
        device="cuda",
        title="Server NVIDIA Tesla A100 GPU (Static Shapes)")
    df = df.append(summary_df, ignore_index=True)
    summary_html += html

  # Generate Server CPU Dynamic.
  if args.cpu_iree_csv is not None:
    summary_df, html = generate_table(
        args.timestamp,
        args.cpu_iree_csv,
        args.cpu_baseline_csv,
        shark_csv=args.cpu_shark_csv,
        shape_type="dynamic",
        device="cpu",
        title="Server Intel Ice Lake CPU (Dynamic Shapes)")
    df = df.append(summary_df, ignore_index=True)
    summary_html += html

  # Generate Server GPU Dynamic.
  if args.gpu_iree_csv is not None:
    summary_df, html = generate_table(
        args.timestamp,
        args.gpu_iree_csv,
        args.gpu_baseline_csv,
        shark_csv=args.gpu_shark_csv,
        shape_type="dynamic",
        device="cuda",
        title="Server NVIDIA Tesla A100 GPU (Dynamic Shapes)")
    df = df.append(summary_df, ignore_index=True)
    summary_html += html

  df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

  trends_df, trends_html = generate_trends(args.timestamp, df)
  trends_df.to_csv(os.path.join(args.output_dir, "trends.csv"), index=False)

  main_html += trends_html
  main_html += summary_html

  html_path = pathlib.Path(os.path.join(args.output_dir, "summary.html"))
  html_path.write_text(main_html)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--timestamp",
      type=str,
      default=None,
      help="Timestamp for when the results were captured.")
  parser.add_argument(
      "--cpu_shark_csv",
      type=str,
      default=None,
      help="The path to the csv file with CPU benchmarking results from the "
      "SHARK runtime.")
  parser.add_argument(
      "--cpu_iree_csv",
      type=str,
      default=None,
      help="The path to the csv file with CPU benchmarking results from IREE.")
  parser.add_argument(
      "--cpu_baseline_csv",
      type=str,
      default=None,
      help="The path to the csv file containing baseline CPU results.")
  parser.add_argument(
      "--gpu_shark_csv",
      type=str,
      default=None,
      help="The path to the csv file with GPU benchmarking results from the "
      "SHARK runtime.")
  parser.add_argument(
      "--gpu_iree_csv",
      type=str,
      default=None,
      help="The path to the csv file with CPU benchmarking results from IREE.")
  parser.add_argument(
      "--gpu_baseline_csv",
      type=str,
      default=None,
      help="The path to the csv file containing baseline GPU results.")
  parser.add_argument("--iree_version",
                      type=str,
                      default=None,
                      help="The IREE version.")
  parser.add_argument("--shark_version",
                      type=str,
                      default=None,
                      help="The SHARK version.")
  parser.add_argument(
      "--output_dir",
      type=pathlib.Path,
      default="/tmp/shark_summary",
      help="The path to the output html file that summarizes results.")
  return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
