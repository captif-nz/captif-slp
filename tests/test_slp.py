
import pandas as pd
import numpy as np
from datetime import date
from numpy import nan

from captif_slp.slp import (
    apply_dropout_correction,
    apply_lowpass_filter,
    apply_slope_correction,
    apply_spike_removal,
    build_resampled_trace,
    calculate_msd,
    calculate_trace_sample_spacing,
    extract_segment_traces_from_trace,
    load_reading,
    dropout_correction_start_end,
    dropout_correction_interpolate,
    Reading,
    Segment,
)


def test_load_reading_4102a5dd(data_path):
    path = data_path.joinpath("4102a5dd.dat")
    meta, trace = load_reading(path)
    assert meta == {
        "date": date(2019, 1, 16),
        "file_number": 0,
    }
    assert np.array_equal(
        trace["relative_height_mm"], [nan, nan, nan, nan, nan, -0.122, -0.065], True,
    )


def test_reading_sample_spacing_4102a5dd(data_path):
    trace = pd.DataFrame({"distance_mm": [0, 0.2, 0.4, 0.6, 0.8]})
    assert calculate_trace_sample_spacing(trace) == 0.2


def test_dropout_correction_start_end():
    trace = pd.DataFrame([
        {"distance_mm": 0, "relative_height_mm": nan},
        {"distance_mm": 0.1, "relative_height_mm": 0.1},
        {"distance_mm": 0.3, "relative_height_mm": nan},
        {"distance_mm": 0.4, "relative_height_mm": 0.4},
        {"distance_mm": 0.5, "relative_height_mm": nan},
    ])
    trace = dropout_correction_start_end(trace)
    assert np.array_equal(trace['relative_height_mm'], [0.1, 0.1, nan, 0.4, 0.4], True)


def test_dropout_correction_interpolate():
    trace = pd.DataFrame([
        {"distance_mm": 0, "relative_height_mm": nan},
        {"distance_mm": 0.1, "relative_height_mm": 0.1},
        {"distance_mm": 0.3, "relative_height_mm": nan},
        {"distance_mm": 0.4, "relative_height_mm": 0.4},
        {"distance_mm": 0.5, "relative_height_mm": nan},
    ])
    trace = dropout_correction_interpolate(trace)
    assert np.array_equal(trace['relative_height_mm'], [nan, 0.1, 0.3, 0.4, nan], True)


def test_apply_dropout_correction():
    trace = pd.DataFrame([
        {"distance_mm": 0, "relative_height_mm": nan},
        {"distance_mm": 0.1, "relative_height_mm": 0.1},
        {"distance_mm": 0.3, "relative_height_mm": nan},
        {"distance_mm": 0.4, "relative_height_mm": 0.4},
        {"distance_mm": 0.5, "relative_height_mm": nan},
    ])
    trace = apply_dropout_correction(trace)
    assert np.array_equal(trace['relative_height_mm'], [0.1, 0.1, 0.3, 0.4, 0.4], True)


def test_apply_spike_removal_middle():
    trace = pd.DataFrame({
        "distance_mm": [0, 0.1, 0.2, 0.3, 0.4],
        "relative_height_mm": [0.1, 0, 0.3, 0, 0.1],
    })
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(trace["relative_height_mm"], [0.1, 0.1, 0.1, 0.1, 0.1], True)


def test_apply_spike_removal_start_end():
    trace = pd.DataFrame({
        "distance_mm": [0, 0.1, 0.2, 0.3, 0.4],
        "relative_height_mm": [0.3, 0, 0.1, 0, 0.3],
    })
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(trace["relative_height_mm"], [0.1, 0.1, 0.1, 0.1, 0.1], True)


def test_apply_spike_removal():
    trace = pd.DataFrame({
        "distance_mm": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "relative_height_mm": [0.3, 0, 0.2, 0, 0.3, 0, -0.2, 0, 0.3],
    })
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(
        trace["relative_height_mm"],
        [0.2, 0.2, 0.2, 0.1, 0, -0.1, -0.2, -0.2, -0.2],
        equal_nan=True,
    )


def test_apply_spike_removal_no_spikes():
    trace = pd.DataFrame({
        "distance_mm": [0, 0.1, 0.2, 0.3, 0.4],
        "relative_height_mm": [0, 0.1, 0.2, 0.1, -0.1],
    })
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(trace["relative_height_mm"], [0, 0.1, 0.2, 0.1, -0.1], True)


def test_apply_lowpass_filter():
    trace = pd.DataFrame({
        "relative_height_mm": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })
    trace = apply_lowpass_filter(trace, sample_spacing_mm=0.5)
    assert np.array_equal(
        trace["relative_height_mm"],
        [
            0.9999999586197605,
            0.2916296316193974,
            -0.05016417032052734,
            -0.07035465905427346,
            -0.012187621988978041,
            0.009500958448235173,
            0.005199674115851917,
            -0.00021162041710998092,
            -0.0009839093400297305,
            -5.029346882672568e-05,
        ]
    )


def test_build_resampled_trace():
    trace = pd.DataFrame({
        "distance_mm": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
        "relative_height_mm": [0.3, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, -0.1, 0.1, 0.1, 0.5],
    })
    resampled_trace = build_resampled_trace(trace, target_sample_spacing_mm=0.5)

    assert np.array_equal(resampled_trace["distance_mm"], [0.5, 1, 1.5, 2, 2.5])
    assert np.array_equal(resampled_trace["relative_height_mm"], [0.2, 0.15, 0.15, 0, 0.3])


def test_extract_segment_traces_from_trace():
    trace = pd.DataFrame({
        "distance_mm": [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "relative_height_mm": [0.3, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, -0.1, 0.1, 0.1, 0.5],
    })
    segment_traces = list(extract_segment_traces_from_trace(trace, segment_length_mm=100))

    assert len(segment_traces) == 5
    assert [len(tt) for tt in segment_traces] == [3, 2, 2, 2, 2]
    assert np.array_equal(segment_traces[1]["distance_mm"], [150, 200])
    assert np.array_equal(segment_traces[1]["relative_height_mm"], [0.2, 0.1])


def test_apply_slope_correction():
    trace = pd.DataFrame({
        "distance_mm": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
        "relative_height_mm": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })
    slope_suppressed_trace = apply_slope_correction(trace)
    assert np.array_equal(slope_suppressed_trace["relative_height_mm"], [0] * 11)


def test_calculate_msd():
    trace = pd.DataFrame({
        "distance_mm": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "relative_height_mm": [1, 0, 0, 0, 0, 2, 0, 0, 0, 0,],
    })
    assert calculate_msd(trace) == ((1 + 2) / 2) - (3/10)

class TestReading:

    def test_from_trace(self):
        trace = pd.DataFrame({
            "distance_mm": np.arange(0, 10, 0.25),
            "relative_height_mm": [0.1] * 40,
        })
        reading = Reading.from_trace(trace)

        pd.testing.assert_frame_equal(reading.trace, trace, check_dtype=False)
        assert len(reading.resampled_trace) == 20

    def test_from_file_4102a5dd(self):
        pass

    def test_segments(self):
        pass


class TestSegment:

    def test_dropout_ratio(self):
        trace = pd.DataFrame({"dropout": [True, False, True, False, False]})
        segment = Segment(segment_no=1, trace=trace, resampled_trace=pd.DataFrame())
        assert segment.dropout_ratio == 2/5

    def test_spike_ratio(self):
        resampled_trace = pd.DataFrame({"spike": [False, False, True, False, False]})
        segment = Segment(segment_no=1, trace=pd.DataFrame(), resampled_trace=resampled_trace)
        assert segment.spike_ratio == 1/5

    def test_is_valid(self):
        pass


