import os
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
):
    limit_cpu()
    results = {}
    for path in paths:
        result = Reading.from_file(
            path,
            segment_length_mm=segment_length_mm,
            target_sample_spacing_mm=target_sample_spacing_mm,
            evaluation_length_m=evaluation_length_m,
            parallel=False
        ).mpd()
        print(result)
        results[path] = result
    return results


def process_files(
    paths: List[Union[str, Path]],
    segment_length_mm: int = 100,
    target_sample_spacing_mm: float = 0.5,
    evaluation_length_m: Optional[float] = None,
):
    tasks = [
        _process_files(
            pp.tolist(),
            segment_length_mm=segment_length_mm,
            target_sample_spacing_mm=target_sample_spacing_mm,
            evaluation_length_m=evaluation_length_m,
        ) for pp in np.array_split(paths, CPU_COUNT)
    ]
    results = [tt.result() for tt in tasks]
    return {kk: vv for rr in results for kk, vv in rr.items()}
