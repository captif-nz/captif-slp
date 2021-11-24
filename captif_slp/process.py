import os
import psutil
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        result = Reading.from_file(
            path,
            segment_length_mm=segment_length_mm,
            target_sample_spacing_mm=target_sample_spacing_mm,
            evaluation_length_m=evaluation_length_m,
            parallel=False,
            alpha=alpha,
            start_mm=start_mm,
            end_mm=end_mm,
            detect_plates=detect_plates,
        ).mpd(include_meta=True)
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
    except:
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
        ) for pp in np.array_split(paths, CPU_COUNT)
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
    path: Union[str, Path]
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


@unsync(cpu_bound=True)
def _check_files(
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
    for path in paths:
        path = Path(path)

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
        for segment in reading.segments:
            plotpath = path.with_name(f"{path.stem}_{segment.segment_no}.png")
            labview_segment_path = (
                path.parent.joinpath(path.stem + "_labview_segments", plotpath.name)
                .with_suffix(".dat")
            )

            fig, ax = plt.subplots(3, 1, figsize=(10, 12))
            ax[0].plot(
                segment.trace["distance_mm"],
                segment.trace["relative_height_mm"],
                "C0-",
                label="raw trace (with dropout corr.) ",
            )
            is_dropout = segment.trace["dropout"]
            ax[0].plot(
                segment.trace.loc[is_dropout, "distance_mm"],
                segment.trace.loc[is_dropout, "relative_height_mm"],
                "C0x",
                label=f"dropouts ({is_dropout.mean()*100:.1f}%)",
            )
            ax[0].plot(
                segment.resampled_trace["distance_mm"],
                segment.resampled_trace["relative_height_mm_no_spike_correction"],
                "C1-",
                label="resampled trace",
            )
            is_spike = segment.resampled_trace["spike"]
            ax[0].plot(
                segment.resampled_trace.loc[is_spike, "distance_mm"],
                segment.resampled_trace.loc[is_spike, "relative_height_mm_no_spike_correction"],
                "rx",
                label=f"spikes ({is_spike.mean()*100:.1f}%)",
            )

            ax[1].plot(
                segment.resampled_trace["distance_mm"],
                segment.resampled_trace["relative_height_mm_no_lowpass_filter"],
                "C0-",
                label="resampled trace (with spike corr.)",
            )
            ax[1].plot(
                segment.resampled_trace["distance_mm"],
                segment.resampled_trace["relative_height_mm_no_slope_correction"],
                "C1-",
                label="resampled trace (with lowpass filter)",
            )
            ax[1].plot(
                segment.resampled_trace["distance_mm"],
                segment.resampled_trace["slope_correction"],
                "C1--",
                label="slope corr.",
            )
            ax[1].plot(
                segment.resampled_trace["distance_mm"],
                segment.resampled_trace["relative_height_mm"],
                "k-",
                label="resampled trace (with slope corr.)",
            )

            n_samples = len(segment.resampled_trace)
            i_midpoint = int(np.ceil(n_samples / 2))
            peak1 = segment.resampled_trace.iloc[:i_midpoint]["relative_height_mm"].max()
            peak2 = segment.resampled_trace.iloc[i_midpoint:]["relative_height_mm"].max()
            peak_average = (peak1 + peak2) / 2
            profile_average = segment.resampled_trace["relative_height_mm"].mean()

            ax[2].plot(
                segment.resampled_trace.iloc[:i_midpoint]["distance_mm"],
                [peak1]*len(segment.resampled_trace.iloc[:i_midpoint]),
                "r--",
                label=f"peak level 1 (Python)",
            )

            ax[2].plot(
                segment.resampled_trace.iloc[i_midpoint:]["distance_mm"],
                [peak2]*len(segment.resampled_trace.iloc[i_midpoint:]),
                "b--",
                label=f"peak level 2 (Python)",
            )

            ax[2].plot(
                segment.resampled_trace["distance_mm"],
                [peak_average]*len(segment.resampled_trace),
                "k--",
                label=f"average peak level (Python)",
            )

            ax[2].plot(
                segment.resampled_trace["distance_mm"],
                segment.resampled_trace["relative_height_mm"],
                "k-",
                label=f"Python trace ({peak_average:.3f} mm) (valid={segment.is_valid})",
            )

            if labview_segment_path.exists():
                df = pd.read_csv(labview_segment_path, sep="\t", header=None)
                x = df.loc[0].to_numpy()
                y = df.loc[1].to_numpy()

                n_samples = len(x)
                i_midpoint = int(np.ceil(n_samples / 2))
                peak1 = y[:i_midpoint].max()
                peak2 = y[i_midpoint:].max()
                peak_average = (peak1 + peak2) / 2
                profile_average = y.mean()

                ax[2].plot(x, y, "C0-", label=f"LabVIEW trace ({peak_average-profile_average:.3f} mm)")

            for aa in ax:
                aa.legend(loc="lower right")
                aa.grid()
                aa.set_xlim(
                    segment.segment_no*segment_length_mm,
                    (segment.segment_no + 1)*segment_length_mm
                )
                aa.set_ylim(-10, 10)
            fig.savefig(plotpath)
            plt.close(fig)

    return None


def check_files(
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
    except:
        pass

    tasks = [
        _check_files(
            pp.tolist(),
            segment_length_mm=segment_length_mm,
            target_sample_spacing_mm=target_sample_spacing_mm,
            evaluation_length_m=evaluation_length_m,
            alpha=alpha,
            start_mm=start_mm,
            end_mm=end_mm,
            detect_plates=detect_plates,
        ) for pp in np.array_split(paths, CPU_COUNT)
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
