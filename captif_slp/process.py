import csv
import os
import pandas as pd
import psutil
import multiprocessing
import numpy as np
from pathlib import Path
from platform import system
from typing import List, Optional, Union
from unsync import unsync
import  json

from dataclasses import dataclass
from rich.live import Live

from .slp import Reading


CPU_COUNT = multiprocessing.cpu_count()
OS = system()

@dataclass
class CountMonitoring:
    total_samples: int = None
    current_sample: int = None

    def __repr__(self):
        return (
            f"Total sample count: {self.total_samples}\n"
            f"Current sample being processed: {self.current_sample}\n"
            f"Percentage complete: {self.current_sample/self.total_samples*100:.2f}%")

def limit_cpu():
    p = psutil.Process(os.getpid())
    if OS == "Windows":
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        p.nice(19)


@unsync(cpu_bound=True)
def _process_files(
    paths: List[Union[str, Path]],
    segment_length_mm: int = 100,
    target_sample_spacing_mm: float = 0.5,
    evaluation_length_m: Optional[float] = None,
    alpha: int = 3,
    start_mm: Optional[float] = None,
    end_mm: Optional[float] = None,
    detect_plates: bool = False,
):
    limit_cpu()
    results = {}
    for path in paths:
        reading = Reading.from_file(
            path,
            segment_length_mm=segment_length_mm,
            target_sample_spacing_mm=target_sample_spacing_mm,
            evaluation_length_m=evaluation_length_m,
            parallel=False,
            alpha=alpha,
            start_mm=start_mm,
            end_mm=end_mm,
            detect_plates=detect_plates,
        )
        result, trace = reading.result()
        result[
            "trace"
        ] = trace  # TODO: fix this for when evaluation_length_m is not None
        results[path] = result

    return results


def process_generic_files(
    paths: List[Union[str, Path]],
    segment_length_mm: int = 100,
    target_sample_spacing_mm: float = 0.5,
    evaluation_length_m: Optional[float] = None,
    alpha: int = 3,
    start_mm: Optional[float] = None,
    end_mm: Optional[float] = None,
    detect_plates: bool = False,
):
    try:
        paths = [pp.as_posix() for pp in paths]
    except Exception:
        pass

    tasks = [
        _process_files(
            pp.tolist(),
            segment_length_mm=segment_length_mm,
            target_sample_spacing_mm=target_sample_spacing_mm,
            evaluation_length_m=evaluation_length_m,
            alpha=alpha,
            start_mm=start_mm,
            end_mm=end_mm,
            detect_plates=detect_plates,
        )
        for pp in np.array_split(paths, CPU_COUNT)
    ]
    results = [tt.result() for tt in tasks]
    results = {kk: vv for rr in results for kk, vv in rr.items()}
    results_ = []
    for kk, vv in results.items():
        pp = Path(kk)
        vv["folder"] = pp.parent.absolute().as_posix()
        vv["filename"] = pp.name
        results_.append(vv)
    return results_


def process_files(
    path: Union[str, Path],
):
    paths = list(Path(path).glob("*.dat"))
    return process_generic_files(
        paths=paths,
        segment_length_mm=100,
        target_sample_spacing_mm=0.5,
        evaluation_length_m=None,
        alpha=6,
        detect_plates=True,
    )


# def read_traces_from_transverse_file(path):
#     timestamps, traces = [], []
#     with open(path, 'r') as f:
#         reader = csv.reader(f)
#         trace = {}
#         for i, row in enumerate(reader):
#             # timestamps.append(row[0])
#             temp_timestamp = row[0]
#             row = row[1:]  # drop timestamp
#             if i % 2 == 0:
#                 trace['distance_mm'] = [float(val) for val in row]
#                 # timestamps.append(temp_timestamp)
#             else:
#                 temp_trace = [float(val) for val in row]
#                 # Check if all elements in temp_trace are NaN
#                 if not all(np.isnan(temp_trace)):
#                         # continue
#                     # trace['relative_height_mm'] = [float(val) for val in row]
#                     trace['relative_height_mm'] = temp_trace
#                     traces.append(trace)
#                     timestamps.append(temp_timestamp)
#                     trace = {}
#     return timestamps, [pd.DataFrame(tt) for tt in traces]


def process_transverse_file(
    path: Union[str, Path],
    segment_length_mm: int = 100,
    target_sample_spacing_mm: float = 0.5,
    evaluation_length_m: Optional[float] = None,
    alpha: int = 3,
    start_mm: Optional[float] = -75, #-50, Change to -75
    end_mm: Optional[float] = 75, # 50, Change to 75
    detect_plates: bool = False,
):

    timestamps, traces = read_traces_from_transverse_file(path)

    results = []
    ndx = 0
    count_monitoring = CountMonitoring(total_samples=len(timestamps))
    with Live(auto_refresh=False) as live:
        print("Starting Processing")
        for timestamp, trace in zip(timestamps, traces):
            reading = Reading.from_trace(
                trace,
                segment_length_mm=segment_length_mm,
                target_sample_spacing_mm=target_sample_spacing_mm,
                evaluation_length_m=evaluation_length_m,
                alpha=alpha,
                start_mm=start_mm,
                end_mm=end_mm,
                detect_plates=detect_plates,
                trace_type = "transverse",
            )
            result, trace = reading.result()
            result["trace"] = trace
            result["timestamp"] = timestamp
            results.append(result)
            ndx+=1
            count_monitoring.current_sample = ndx
            live.update(str(count_monitoring))
            live.refresh()

    return results

def extract_transverse_profile(row):
    z_offset = row["z_offset_um"] / 1000.0
    z_scale = row["z_scale_nm"] / 1000000.0
    profile_data = np.asarray(json.loads(row["profile"]))
    profile_data = profile_data.astype(np.double)
    profile_data[profile_data==-32768] = np.nan
    profile_data = (profile_data * z_scale) + z_offset
    return profile_data

def extract_transverse_x_arr(row):
    x_offset = row["x_offset_um"] / 1000.0
    x_scale = row["x_scale_nm"] / 1000000.0
    x_width = row["width"]
    x_arr = (np.asarray(range(x_width), dtype=np.double) * x_scale) + x_offset
    return x_arr


def read_traces_from_transverse_file(path):
    # Read in the data
    df = pd.read_csv(path, header=0)
    # df = df.drop(df.index[0:10]) # Check the validity of this step

    # Only keep every 100th row
    # df = df.iloc[::100, :]
    # Internal clock differences
    df["int_diff"] = df["internal_timestamp"].diff()
    df["int_diff"].iloc[0] = 0
    df["int_cumsum"] = df["int_diff"].cumsum()
    # Cumulative sum of internal clock differences
    # df["nzt_datetime_corrected"] = pd.to_datetime(df["utc_timestamp"]/1e6, unit='s', utc=True).map(lambda x: x.tz_convert('Pacific/Auckland'))
    df["corrected_utc_timestamp"] = df["utc_timestamp"]
    # df["corrected_utc_timestamp"] = df["utc_timestamp"].iloc[0]  + df["int_cumsum"] # This isn't correct...
    df["nzt_datetime_corrected"] = pd.to_datetime(df["corrected_utc_timestamp"]/1e6, unit='s', utc=True).map(lambda x: x.tz_convert('Pacific/Auckland'))
    # df["nzt_datetime_corrected"] = pd.to_datetime(df["corrected_utc_timestamp"]/1e6, unit='s', utc=True).map(lambda x: x.tz_convert('Pacific/Auckland'))
    df["z_profile"] = df.apply(extract_transverse_profile, axis=1)
    df["x_arr"] = df.apply(extract_transverse_x_arr, axis=1)

    # This is done to match the other transverse processing
    traces = []
    timestamps = []
    for row in df.itertuples():
        trace = {}
        trace['distance_mm'] = row.x_arr
        trace['relative_height_mm'] = row.z_profile
        traces.append(trace)
        timestamps.append(row.nzt_datetime_corrected)

    # timestamps = df["nzt_datetime_corrected"].tolist()
    
    return timestamps, [pd.DataFrame(tt) for tt in traces]