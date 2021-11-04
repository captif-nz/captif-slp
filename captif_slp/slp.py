
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from captif_data_structures.readers import TextureReader
from scipy.signal.signaltools import sosfiltfilt

from .signal import build_highpass_filter, build_lowpass_filter


@dataclass
class Segment:
    segment_no: int
    trace: pd.DataFrame
    resampled_trace: pd.DataFrame
    segment_length_mm: int = 100
    resampled_sample_spacing_mm: float = 0.5
    evaluation_length_position_m: Optional[float] = None

    @property
    def dropout_ratio(self) -> float:
        """Dropout ratio for the segment according to section 7.3 of ISO 13473-1:2019."""
        return self.trace["dropout"].mean()

    @property
    def spike_ratio(self) -> float:
        """Spike ratio for the segment according to section 7.5 of ISO 13473-1:2019."""
        return self.resampled_trace["spike"].mean()

    @property
    def msd(self) -> Optional[float]:
        """
        Mean segment depth (MSD) in millimetres according to section 7.8 of
        ISO 13473-1:2019.

        Has a value of None if the segment is invalid due to a high number of dropouts or
        spikes.

        """
        if not self.is_valid:
            return None
        return calculate_msd(self.resampled_trace)

    @property
    def is_valid(self) -> bool:
        """
        Segment validity (True/False) based on the dropout ratio and spike ratio
        (sections 7.3 and 7.5 of ISO 13473-1:2019).

        """
        if self.dropout_ratio > 0.1:
            return False
        if self.spike_ratio > 0.05:
            return False
        # TODO: check start/end dropout correction does not exceed 5 mm
        return True


@dataclass
class Reading:
    meta: Optional[dict]
    trace: pd.DataFrame
    resampled_trace: pd.DataFrame
    segment_length_mm: int
    resampled_sample_spacing_mm: float
    evaluation_length_m: Optional[float] = None

    @classmethod
    def from_trace(
        cls,
        trace,
        meta=None,
        segment_length_mm: int = 100,
        target_sample_spacing_mm: float = 0.5,
        evaluation_length_m: Optional[float] = None,
    ):
        trace = append_dropout_column(trace)
        trace = apply_dropout_correction(trace)

        resampled_trace = build_resampled_trace(trace, target_sample_spacing_mm)
        resampled_trace = apply_spike_removal(resampled_trace)

        if evaluation_length_m is None:
            resampled_trace = apply_slope_correction(resampled_trace)
        else:
            resampled_trace = apply_highpass_filter(resampled_trace, target_sample_spacing_mm)

        resampled_trace = apply_lowpass_filter(resampled_trace, target_sample_spacing_mm)

        return Reading(
            meta, trace, resampled_trace, segment_length_mm, target_sample_spacing_mm,
            evaluation_length_m,
        )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        segment_length_mm: int = 100,
        target_sample_spacing_mm: float = 0.5,
        evaluation_length_m: Optional[float] = None,
        parallel: bool = True,
    ):
        meta, trace = load_reading(path, parallel=parallel)
        return cls.from_trace(
            trace, meta, segment_length_mm, target_sample_spacing_mm, evaluation_length_m,
        )

    @property
    def segments(self):
        traces = zip(
            extract_segment_traces_from_trace(self.trace, self.segment_length_mm),
            extract_segment_traces_from_trace(self.resampled_trace, self.segment_length_mm),
        )
        for ii, (segment_trace, resampled_segment_trace) in enumerate(traces):

            evaluation_length_position_m = calculate_evaluation_length_position(
                segment_trace["distance_mm"].min(), self.evaluation_length_m)
                
            yield Segment(
                segment_no=ii,
                trace=segment_trace,
                resampled_trace=resampled_segment_trace,
                segment_length_mm=self.segment_length_mm,
                resampled_sample_spacing_mm=self.resampled_sample_spacing_mm,
                evaluation_length_position_m=evaluation_length_position_m,
            )

    def msd(self) -> List[dict]:
        """Mean segment depths (MSD) for the segments making up the profile."""
        return [
            {
                "msd": ss.msd,
                "evaluation_length_position_m": ss.evaluation_length_position_m,
            }
            for ss in self.segments
        ]

    def mpd(self) -> Union[dict, pd.DataFrame]:
        """Mean profile depth (MPD) results for each evaluation length."""
        df = pd.DataFrame.from_records(self.msd())
        results = []
        for distance_m, gg in df.groupby("evaluation_length_position_m", dropna=False):
            valid_segments = (~gg["msd"].isnull()).sum()
            valid_segment_ratio = valid_segments / len(gg)
            results.append(
                {
                    "distance_m": distance_m,
                    "mean": gg["msd"].mean(),
                    "stdev": gg["msd"].std(),
                    "valid_segments": valid_segments,
                    "valid_segment_ratio": valid_segment_ratio,
                    "is_valid": valid_segment_ratio >= 0.5,
                }
            )
        return results[0] if len(results) == 1 else pd.DataFrame(results)


def extract_segment_traces_from_trace(trace: pd.DataFrame, segment_length_mm: int):
    bins = generate_trace_bins(trace, segment_length_mm)
    yield from (
        tt for _, tt in trace.groupby(
            pd.cut(trace["distance_mm"], bins, include_lowest=True)
        )
    )


def generate_trace_bins(trace: pd.DataFrame, bin_width_mm: float):
    return np.arange(0, trace["distance_mm"].max() + bin_width_mm, bin_width_mm)


def build_resampled_trace(trace: pd.DataFrame, target_sample_spacing_mm: float):
    if calculate_trace_sample_spacing(trace) == target_sample_spacing_mm:
        return trace.copy()

    trace["group"] = np.ceil(trace["distance_mm"] / target_sample_spacing_mm)
    g0 = trace.loc[0, "group"]
    trace.loc[0, "group"] = 1 if g0 == 0 else g0

    resampled_trace = trace[["group", "relative_height_mm"]].groupby("group").mean()
    resampled_trace["distance_mm"] = resampled_trace.index * target_sample_spacing_mm
    resampled_trace.reset_index(drop=True, inplace=True)
    trace.drop(columns=["group"], inplace=True)

    resampled_trace["relative_height_mm"] = resampled_trace["relative_height_mm"].round(6)
    return resampled_trace


def load_reading(path: Union[str, Path], parallel: bool = True):
    meta, table_rows, _ = TextureReader.load(path, parallel=parallel)
    trace = pd.DataFrame(table_rows).sort_values("distance_mm").reset_index(drop=True)
    return meta, trace


def append_dropout_column(trace: pd.DataFrame):
    trace["dropout"] = trace["relative_height_mm"].isnull()
    return trace


def apply_dropout_correction(trace: pd.DataFrame):
    if trace["relative_height_mm"].isnull().sum() == 0:
        return trace
    trace = dropout_correction_start_end(trace)
    trace = dropout_correction_interpolate(trace)
    trace["relative_height_mm"] = trace["relative_height_mm"].round(6)
    return trace


def calculate_trace_sample_spacing(trace: pd.DataFrame) -> float:
    return trace["distance_mm"].diff().mean()


def apply_spike_removal(trace: pd.DataFrame, alpha: float = 3):
    threshold = round(alpha * calculate_trace_sample_spacing(trace), 6)
    ss = (trace["relative_height_mm"].diff().abs() >= threshold).to_numpy()[1:]

    trace["spike"] = (
        np.insert(ss, 0, False) |  # spikes in forward direction
        np.append(ss, False)  # spikes in reverse direction
    )
    trace.loc[trace["spike"], "relative_height_mm"] = None
    trace = apply_dropout_correction(trace)
    return trace


def apply_slope_correction(trace: pd.DataFrame):
    p = np.polyfit(trace["distance_mm"], trace["relative_height_mm"], deg=1)
    slope_correction = pd.Series(
        trace["distance_mm"] * p[0] + p[1], index=trace.index,
    )
    trace["relative_height_mm"] -= slope_correction
    trace["relative_height_mm"] = trace["relative_height_mm"].round(6)
    return trace


def apply_lowpass_filter(trace: pd.DataFrame, sample_spacing_mm: float):
    sos = build_lowpass_filter(sample_spacing_mm)
    trace["relative_height_mm"] = sosfiltfilt(sos, trace["relative_height_mm"])
    return trace


def apply_highpass_filter(trace: pd.DataFrame, sample_spacing_mm: float):
    sos = build_highpass_filter(sample_spacing_mm)
    trace["relative_height_mm"] = sosfiltfilt(sos, trace["relative_height_mm"])
    return trace


def dropout_correction_start_end(trace: pd.DataFrame):
    yy = trace["relative_height_mm"].copy()
    valid_index = yy.loc[~yy.isna()].index

    # Fill start of trace if it contains dropouts:
    if np.isnan(yy.iloc[0]):
        yy.loc[:valid_index[0]] = yy.loc[valid_index[0]]

    # Fill end of trace if it contains dropouts:
    if np.isnan(yy.iloc[-1]):
        yy.loc[valid_index[-1]:] = yy.loc[valid_index[-1]]

    trace["relative_height_mm"] = yy
    return trace


def dropout_correction_interpolate(trace: pd.DataFrame):
    return (trace
        .set_index("distance_mm", drop=True)  # so distance weighing can be used in interpolation
        .interpolate(method="index", limit_area="inside")
        .reset_index(drop=False)  # move distance back to a column
    )


def calculate_msd(trace: pd.DataFrame) -> float:
    n_samples = len(trace)
    i_midpoint = int(np.ceil(n_samples / 2))
    peak1 = trace.iloc[:i_midpoint]["relative_height_mm"].max()
    peak2 = trace.iloc[i_midpoint:]["relative_height_mm"].max()
    peak_average = (peak1 + peak2) / 2
    profile_average = trace["relative_height_mm"].mean()
    return peak_average - profile_average


def calculate_evaluation_length_position(
    segment_start_position_mm: float,
    evaluation_length_m: Optional[float] = None
) -> float:
    if evaluation_length_m is None:
        return None

    position_no = int(np.floor(segment_start_position_mm / (evaluation_length_m * 1000)))
    return (position_no + 1) * evaluation_length_m
