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

from .slp import Reading


CPU_COUNT = multiprocessing.cpu_count()
OS = system()


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


def read_traces_from_transverse_file(path):
    timestamps, traces = [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        trace = {}
        for i, row in enumerate(reader):
            timestamps.append(row[0])
            row = row[1:]  # drop timestamp
            if i % 2 == 0:
                trace['distance_mm'] = [float(val) for val in row]
            else:
                temp_trace = [float(val) for val in row]
                # Check if all elements in temp_trace are NaN
                if all(np.isnan(temp_trace)):
                    continue
                # trace['relative_height_mm'] = [float(val) for val in row]
                trace['relative_height_mm'] = temp_trace
                traces.append(trace)
                trace = {}
    return timestamps, [pd.DataFrame(tt) for tt in traces]


def process_transverse_file(
    path: Union[str, Path],
    segment_length_mm: int = 100,
    target_sample_spacing_mm: float = 0.5,
    evaluation_length_m: Optional[float] = None,
    alpha: int = 3,
    start_mm: Optional[float] = -50,
    end_mm: Optional[float] = 50,
    detect_plates: bool = False,
):

    timestamps, traces = read_traces_from_transverse_file(path)

    results = []
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
        )
        result, trace = reading.result()
        result["trace"] = trace
        result["timestamp"] = timestamp
        results.append(result)

    return results