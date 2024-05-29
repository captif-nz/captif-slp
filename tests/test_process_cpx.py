from datetime import datetime
from itertools import chain
import multiprocessing
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from pytest import mark

from captif_slp import slp


TEXTURE_PROFILE_RANGE = (-25, 25)
TEXTURE_ALPHA = 6
TEXTURE_ALLOWED_DROPOUT_PERCENT = 0.2
TEXTURE_DIVIDE_SEGMENTS = False


def convert_raw_laser_profile_to_trace(raw: dict):
    """Convert a raw laser reading into a 'trace' as used by the captif-slp
    package.

    Parameters
    ----------
    raw : dict
        Dict containing the raw laser reading.

    Returns
    -------
    dict
        Dict containing the converted laser reading as a captif-slp style
        trace.
    """
    z = raw["profile"].astype(np.double)
    z[z == -32768] = np.nan
    z *= raw["z_scale_nm"] / 1000000.0
    z += raw["z_offset_um"] / 1000.0

    x = np.arange(raw["width"], dtype=np.double)
    x *= raw["x_scale_nm"] / 1000000.0
    x += raw["x_offset_um"] / 1000.0

    df = pd.DataFrame({"relative_height_mm": z}, index=x)
    df.index.name = "distance_mm"
    return df


def trim_trace(trace: pd.DataFrame, start_mm: float, end_mm: float):
    """Trim a trace to the specified start and end distances.

    Parameters
    ----------
    trace : pd.DataFrame
        Trace to trim.
    start_mm : float
        Start distance in millimetres.
    end_mm : float
        End distance in millimetres.

    Returns
    -------
    pd.DataFrame
        Trimmed trace.
    """
    return trace.loc[
        (trace.index.values >= start_mm) & (trace.index.values <= end_mm)
    ].copy()


def check_trace_valid(trace: pd.DataFrame):
    """Check that a trace contains at least one valid value.

    Parameters
    ----------
    trace : pd.DataFrame
        Trace to check.

    Returns
    -------
    bool
        True if the trace contains valid data, False otherwise.
    """
    return not np.isnan(trace["relative_height_mm"].values).all()


def calculate_msd_for_trace(trace: pd.DataFrame):
    trace = trim_trace(trace, *TEXTURE_PROFILE_RANGE)
    if not check_trace_valid(trace):
        return None

    try:
        return slp.Reading.from_trace(
            trace=trace,
            segment_length_mm=TEXTURE_PROFILE_RANGE[1] - TEXTURE_PROFILE_RANGE[0],
            alpha=TEXTURE_ALPHA,
            start_mm=TEXTURE_PROFILE_RANGE[0],
            end_mm=TEXTURE_PROFILE_RANGE[1],
            allowed_dropout_percent=TEXTURE_ALLOWED_DROPOUT_PERCENT,
            divide_segments=TEXTURE_DIVIDE_SEGMENTS,
        ).msd()[0]
    except Exception:
        pass


def process_subset(func_args):
    laser_path, laser_start_sample, laser_end_sample = func_args
    with open(laser_path, "rb") as f:
        _ = pickle.load(f)  # skip the first profile
        size = f.tell()
        f.seek(size * laser_start_sample)
        n_readings = laser_end_sample - laser_start_sample
        results = []
        for _ in range(n_readings):
            try:
                raw_reading = pickle.load(f)
                trace = convert_raw_laser_profile_to_trace(raw_reading)
                results.append(calculate_msd_for_trace(trace))
            except EOFError:
                pass
    return results


@mark.slow
def test_process_cpx_laser_file(data_path):
    laser_path = data_path / "cpx_profiles" / "01_laser_lwp_0.pkl"

    records = 16000
    processes = 16

    subsets = [
        (i * records // processes, (i + 1) * records // processes)
        for i in range(processes)
    ]

    func_args = [(laser_path, start, end) for start, end in subsets]

    with multiprocessing.Pool(processes) as pool:
        results = pool.map(process_subset, func_args)

    results = list(chain(*results))
    assert len(results) == records


if __name__ == "__main__":
    t0 = datetime.now()
    test_process_cpx_laser_file(Path("data"))
    print(datetime.now() - t0)
