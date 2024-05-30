import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
from scipy.signal import sosfiltfilt

from .signal import build_highpass_filter, build_lowpass_filter
from .structures import FileStructure


PLATE_THRESHOLD = 30  # Height values above PLATE_THRESHOLD are treated as a plate.
PLATE_BUFFER = 2  # Buffer added/subtracted from the plate location.


@dataclass
class Segment:
    segment_no: int
    trace: pd.DataFrame
    resampled_trace: pd.DataFrame
    segment_length_mm: int = 100
    resampled_sample_spacing_mm: float = 0.5
    evaluation_length_position_m: Optional[float] = None
    allowed_dropout_percent: float = 10
    divide_segment: bool = True

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
        """
        return calculate_msd(
            self.resampled_trace,
            divide_segment=self.divide_segment,
        )

    @property
    def is_valid(self) -> bool:
        """
        Segment validity (True/False) based on the dropout ratio and spike ratio
        (sections 7.3 and 7.5 of ISO 13473-1:2019).

        """
        if self.dropout_ratio > (self.allowed_dropout_percent / 100):
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
    resampled_sample_spacing_mm: float
    alpha: int
    segment_length_mm: Optional[int] = None
    segment_bins: Optional[list] = None
    evaluation_length_m: Optional[float] = None
    start_mm: Optional[float] = None
    end_mm: Optional[float] = None
    detect_plates: bool = False
    allowed_dropout_percent: float = 10
    divide_segments: bool = True

    @classmethod
    def from_trace(
        cls,
        trace,
        meta=None,
        segment_length_mm: int = 100,
        segment_bins: Optional[list] = None,
        target_sample_spacing_mm: float = 0.5,
        evaluation_length_m: Optional[float] = None,
        alpha: int = 3,
        start_mm: Optional[float] = None,
        end_mm: Optional[float] = None,
        detect_plates: bool = False,
        allowed_dropout_percent: float = 10,
        divide_segments: bool = True,
    ):
        if "distance_mm" in trace.columns:
            trace.set_index("distance_mm", inplace=True)

        if segment_bins is not None:
            segment_length_mm = None

        if detect_plates and (start_mm is None) and (end_mm is None):
            start_mm, end_mm = find_plates(trace)

        trace = trim_trace(trace, start_mm, end_mm)
        trace["relative_height_mm_raw_trace"] = trace["relative_height_mm"]

        trace = append_dropout_column(trace)
        trace = apply_dropout_correction(trace)

        resampled_trace = build_resampled_trace(trace, target_sample_spacing_mm)

        resampled_trace["relative_height_mm_no_spike_correction"] = resampled_trace[
            "relative_height_mm"
        ]
        resampled_trace = apply_spike_removal(resampled_trace, alpha=alpha)

        resampled_trace["relative_height_mm_no_highpass_filter"] = resampled_trace[
            "relative_height_mm"
        ]
        if evaluation_length_m is not None:
            resampled_trace = apply_highpass_filter(
                resampled_trace, target_sample_spacing_mm
            )

        resampled_trace["relative_height_mm_no_lowpass_filter"] = resampled_trace[
            "relative_height_mm"
        ]
        resampled_trace = apply_lowpass_filter(
            resampled_trace, target_sample_spacing_mm
        )

        return Reading(
            meta=meta,
            trace=trace,
            resampled_trace=resampled_trace,
            resampled_sample_spacing_mm=target_sample_spacing_mm,
            alpha=alpha,
            segment_length_mm=segment_length_mm,
            segment_bins=segment_bins,
            evaluation_length_m=evaluation_length_m,
            start_mm=start_mm,
            end_mm=end_mm,
            detect_plates=detect_plates,
            allowed_dropout_percent=allowed_dropout_percent,
            divide_segments=divide_segments,
        )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        segment_length_mm: int = 100,
        segment_bins: Optional[list] = None,
        target_sample_spacing_mm: float = 0.5,
        evaluation_length_m: Optional[float] = None,
        parallel: bool = True,
        alpha: int = 3,
        start_mm: Optional[float] = None,
        end_mm: Optional[float] = None,
        detect_plates: bool = False,
        allowed_dropout_percent: float = 10,
        divide_segments: bool = True,
    ):
        meta, trace = load_reading(path, parallel=parallel)
        return cls.from_trace(
            trace=trace,
            meta=meta,
            segment_length_mm=segment_length_mm,
            segment_bins=segment_bins,
            target_sample_spacing_mm=target_sample_spacing_mm,
            evaluation_length_m=evaluation_length_m,
            alpha=alpha,
            start_mm=start_mm,
            end_mm=end_mm,
            detect_plates=detect_plates,
            allowed_dropout_percent=allowed_dropout_percent,
            divide_segments=divide_segments,
        )

    @property
    def segments(self):
        segment_data = extract_segment_data(
            trace=self.trace,
            resampled_trace=self.resampled_trace,
            segment_length_mm=self.segment_length_mm,
            segment_bins=self.segment_bins,
        )

        segments_ = []
        for ii, (
            segment_trace,
            resampled_segment_trace,
            segment_length_mm,
        ) in enumerate(segment_data):
            n_points = len(resampled_segment_trace)
            max_points = segment_length_mm / self.resampled_sample_spacing_mm
            if (n_points / max_points) < 0.9:
                continue

            # Apply slope correction if "spot" measurement:
            resampled_segment_trace["relative_height_mm_no_slope_correction"] = (
                resampled_segment_trace["relative_height_mm"]
            )
            if self.evaluation_length_m is None:
                resampled_segment_trace = apply_slope_correction(
                    resampled_segment_trace
                )

            evaluation_length_position_m = calculate_evaluation_length_position(
                segment_trace.index.min(), self.evaluation_length_m
            )

            segments_.append(
                Segment(
                    segment_no=ii,
                    trace=segment_trace,
                    resampled_trace=resampled_segment_trace,
                    segment_length_mm=segment_length_mm,
                    resampled_sample_spacing_mm=self.resampled_sample_spacing_mm,
                    evaluation_length_position_m=evaluation_length_position_m,
                    allowed_dropout_percent=self.allowed_dropout_percent,
                    divide_segment=self.divide_segments,
                )
            )
        return segments_

    def msd(self) -> List[dict]:
        """Mean segment depths (MSD) for the segments making up the profile."""
        results = []
        for ss in self.segments:
            results.append(
                {
                    "segment_no": ss.segment_no,
                    "msd": ss.msd,
                    "valid": ss.is_valid,
                    "evaluation_length_position_m": ss.evaluation_length_position_m,
                }
            )
        return results

    def mpd(self, include_meta: bool = False) -> Union[dict, pd.DataFrame]:
        """Mean profile depth (MPD) results for each evaluation length."""
        df = pd.DataFrame.from_records(self.msd())
        results = []
        for distance_m, gg in df.groupby("evaluation_length_position_m", dropna=False):
            valid_segments = gg["valid"].sum()
            proportion_valid_segments = valid_segments / len(gg)
            result = {
                "distance_m": distance_m,
                "mpd": gg.loc[gg["valid"], "msd"].mean(),
                "stdev": gg.loc[gg["valid"], "msd"].std(),
                "valid_segments": valid_segments,
                "proportion_valid_segments": proportion_valid_segments,
                "is_valid": proportion_valid_segments >= 0.5,
            }
            if include_meta and isinstance(self.meta, dict):
                result = append_meta(result, self.meta)

            results.append(result)

        return results[0] if len(results) == 1 else pd.DataFrame(results)

    def result(self) -> Tuple[Union[dict, pd.DataFrame], List[dict]]:
        """
        Returns the Mean profile depth (MPD) results for each evaluation length as either
        a dict (if no evaluation length) or a DataFrame (if evaluation length provided),
        and the resampled trace as a list of dicts.
        """
        return (
            self.mpd(include_meta=True),
            self.resampled_trace.reset_index()[
                ["distance_mm", "relative_height_mm"]
            ].to_dict("records"),
        )


def append_meta(result, meta):
    for kk, vv in meta.items():
        result[kk] = vv
    return result


def trim_trace(
    trace: pd.DataFrame,
    start_mm: Optional[float] = None,
    end_mm: Optional[float] = None,
):
    if end_mm:
        trace = trace.loc[trace.index < end_mm]

    if start_mm:
        trace = trace.loc[trace.index >= start_mm]
        trace.index -= start_mm

    return trace


def find_plates(trace: pd.DataFrame):
    yy = trace["relative_height_mm"].ffill()
    is_plate = yy > PLATE_THRESHOLD

    if is_plate.sum() == 0:
        return None, None

    diff = is_plate.diff()
    diff.iloc[0] = False

    i_midpoint = int(np.ceil(len(trace) / 2))

    start_mm, end_mm = None, None
    try:
        start_mm = (
            trace.iloc[:i_midpoint].loc[diff.iloc[:i_midpoint]].iloc[-1].name
            + PLATE_BUFFER
        )
    except Exception:
        pass

    try:
        end_mm = (
            trace.iloc[i_midpoint:].loc[diff.iloc[i_midpoint:]].iloc[0].name
            - PLATE_BUFFER
        )
    except Exception:
        pass

    return start_mm, end_mm


def extract_segment_traces_from_trace(trace: pd.DataFrame, segment_bins: list):
    yield from (
        tt
        for _, tt in trace.groupby(
            pd.cut(trace.index, segment_bins, include_lowest=True),
            observed=True,
        )
    )


def extract_segment_data(
    trace: pd.DataFrame,
    resampled_trace: pd.DataFrame,
    segment_length_mm: Optional[int] = None,
    segment_bins: Optional[list] = None,
):
    """
    Extracts segment traces (original and resampled) from the reading and calculated the
    resulting segment length of each segment. The segment data is zipped with each
    element containing (trace, resampled_trace, segment_length_mm) for one segment.

    """
    if segment_bins is None and segment_length_mm is None:
        raise ValueError("at least one of segment_length_m or segment_bins must be set")

    if segment_bins is None:
        segment_bins = generate_trace_bins(trace, segment_length_mm)

    if (
        len(segment_bins) == 2
        and segment_bins[0] <= trace.index.min()
        and segment_bins[1] >= trace.index.max()
    ):
        # Skip segment extraction if only one segment is present. This is
        # about 5x faster for cases involving a single segment.
        return [(trace, resampled_trace, np.diff(segment_bins)[0])]

    return zip(
        extract_segment_traces_from_trace(trace, segment_bins),
        extract_segment_traces_from_trace(resampled_trace, segment_bins),
        np.diff(segment_bins),  # segment_length_m of each segment
    )


def generate_trace_bins(trace: pd.DataFrame, bin_width_mm: float):
    return np.arange(0, trace.index.max() + bin_width_mm, bin_width_mm)


def build_resampled_trace(trace: pd.DataFrame, target_sample_spacing_mm: float):
    if calculate_trace_sample_spacing(trace) == target_sample_spacing_mm:
        return trace.copy()

    groups = np.ceil(trace.index.values / target_sample_spacing_mm)
    groups[0] = 1 if groups[0] == 0 else groups[0]

    df = pd.DataFrame(
        {
            "relative_height_mm": trace["relative_height_mm"]
            .groupby(groups)
            .mean()
            .round(6)
            .values,
        },
        index=np.sort(np.unique(groups) * target_sample_spacing_mm),
    )
    df.index.name = "distance_mm"  # this is faster than using set_index.
    return df


def load_reading(path: Union[str, Path], parallel: bool = True):
    meta, table_rows, _ = FileStructure.read(path, parallel=parallel)
    trace = pd.DataFrame(table_rows).sort_values("distance_mm").reset_index(drop=True)
    return meta, trace


def append_dropout_column(trace: pd.DataFrame):
    trace["dropout"] = np.isnan(trace["relative_height_mm"].values)
    return trace


def apply_dropout_correction(trace: pd.DataFrame):
    if not np.isnan(trace["relative_height_mm"].values).any():
        return trace

    if np.isnan(trace["relative_height_mm"].values[0]) or np.isnan(
        trace["relative_height_mm"].values[-1]
    ):
        trace = dropout_correction_start_end(trace)

    if np.isnan(trace["relative_height_mm"].values).any():
        return dropout_correction_interpolate(trace)

    return trace


def calculate_trace_sample_spacing(trace: pd.DataFrame) -> float:
    return np.diff(trace.index.values).mean()


def append_spike_column(trace: pd.DataFrame, alpha: float = 3):
    threshold = round(alpha * calculate_trace_sample_spacing(trace), 6)
    ss = np.abs(np.diff(trace["relative_height_mm"].values)) >= threshold

    if not ss.any():
        trace["spike"] = False
        return trace

    trace["spike"] = np.insert(ss, 0, False) | np.append(  # spikes in forward direction
        ss, False
    )  # spikes in reverse direction
    return trace


def apply_spike_removal(trace: pd.DataFrame, alpha: float = 3):
    trace = append_spike_column(trace, alpha)
    if not trace["spike"].any():
        return trace
    trace.loc[trace["spike"], "relative_height_mm"] = None
    return apply_dropout_correction(trace)


def apply_slope_correction(trace: pd.DataFrame):
    p = np.polyfit(trace.index.values, trace["relative_height_mm"].values, deg=1)
    trace["slope_correction"] = (trace.index.values * p[0]) + p[1]
    trace["relative_height_mm"] = (
        trace["relative_height_mm"].values - trace["slope_correction"].values
    ).round(6)
    return trace


def apply_lowpass_filter(trace: pd.DataFrame, sample_spacing_mm: float):
    trace["relative_height_mm"] = sosfiltfilt(
        build_lowpass_filter(sample_spacing_mm),
        trace["relative_height_mm"].values,
    )
    return trace


def apply_highpass_filter(trace: pd.DataFrame, sample_spacing_mm: float):
    trace["relative_height_mm"] = sosfiltfilt(
        build_highpass_filter(sample_spacing_mm),
        trace["relative_height_mm"].values,
    )
    return trace


def dropout_correction_start_end(trace: pd.DataFrame):
    valid_records = trace.loc[~np.isnan(trace["relative_height_mm"].values)]
    if np.isnan(trace["relative_height_mm"].values)[0]:
        trace.loc[: valid_records.index[0], "relative_height_mm"] = valid_records[
            "relative_height_mm"
        ].values[0]
    if np.isnan(trace["relative_height_mm"].values)[-1]:
        trace.loc[valid_records.index[-1] :, "relative_height_mm"] = valid_records[
            "relative_height_mm"
        ].values[-1]
    return trace


def dropout_correction_interpolate(trace: pd.DataFrame):
    return trace.interpolate(method="index", limit_area="inside")


def calculate_msd(
    trace: pd.DataFrame,
    divide_segment: bool = True,
) -> float:
    """Calculate the mean segment depth (MSD) for a segment.

    Parameters
    ----------
    trace : pd.DataFrame
        The segment trace.
    divide_segment : bool, optional
        Calculate the MSD by splitting the segment into two halves, by
        default True.

    Returns
    -------
    float
        The mean segment depth (MSD) in millimetres.
    """
    if not divide_segment:
        return (
            trace["relative_height_mm"].values.max()
            - trace["relative_height_mm"].values.mean()
        )
    relative_height_mm = trace["relative_height_mm"].values
    n_samples = len(relative_height_mm)
    i_midpoint = n_samples >> 1
    peak1 = relative_height_mm[:i_midpoint].max()
    peak2 = relative_height_mm[i_midpoint:].max()
    return ((peak1 + peak2) / 2) - relative_height_mm.mean()


def calculate_evaluation_length_position(
    segment_start_position_mm: float, evaluation_length_m: Optional[float] = None
) -> float:
    if evaluation_length_m is None:
        return None

    position_no = int(
        np.floor(segment_start_position_mm / (evaluation_length_m * 1000))
    )
    return (position_no + 1) * evaluation_length_m
