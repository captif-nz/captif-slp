import pandas as pd
import numpy as np
from datetime import datetime
from numpy import nan

from captif_slp.slp import (
    apply_dropout_correction,
    apply_lowpass_filter,
    apply_slope_correction,
    apply_spike_removal,
    build_resampled_trace,
    calculate_msd,
    calculate_trace_sample_spacing,
    extract_segment_data,
    extract_segment_traces_from_trace,
    find_plates,
    load_reading,
    dropout_correction_start_end,
    dropout_correction_interpolate,
    Reading,
    Segment,
)


def test_load_reading_4102a5dd(data_path):
    path = data_path.joinpath("structures", "4102a5dd.dat")
    meta, trace = load_reading(path)
    assert meta == {
        "datetime": datetime(2019, 1, 16),
        "file_number": 0,
    }
    assert np.array_equal(
        trace["relative_height_mm"],
        [nan, nan, nan, nan, nan, -0.122, -0.065],
        True,
    )


def test_reading_sample_spacing_4102a5dd(data_path):
    trace = pd.DataFrame({"distance_mm": [0, 0.2, 0.4, 0.6, 0.8]}).set_index(
        "distance_mm"
    )
    assert calculate_trace_sample_spacing(trace) == 0.2


def test_dropout_correction_start_end():
    trace = pd.DataFrame(
        [
            {"distance_mm": 0, "relative_height_mm": nan},
            {"distance_mm": 0.1, "relative_height_mm": 0.1},
            {"distance_mm": 0.3, "relative_height_mm": nan},
            {"distance_mm": 0.4, "relative_height_mm": 0.4},
            {"distance_mm": 0.5, "relative_height_mm": nan},
        ]
    ).set_index("distance_mm")
    trace = dropout_correction_start_end(trace)
    assert np.array_equal(trace["relative_height_mm"], [0.1, 0.1, nan, 0.4, 0.4], True)


def test_dropout_correction_interpolate():
    trace = pd.DataFrame(
        [
            {"distance_mm": 0, "relative_height_mm": nan},
            {"distance_mm": 0.1, "relative_height_mm": 0.1},
            {"distance_mm": 0.3, "relative_height_mm": nan},
            {"distance_mm": 0.4, "relative_height_mm": 0.4},
            {"distance_mm": 0.5, "relative_height_mm": nan},
        ]
    ).set_index("distance_mm")
    trace = dropout_correction_interpolate(trace)
    assert np.array_equal(trace["relative_height_mm"], [nan, 0.1, 0.3, 0.4, nan], True)


def test_apply_dropout_correction():
    trace = pd.DataFrame(
        [
            {"distance_mm": 0, "relative_height_mm": nan},
            {"distance_mm": 0.1, "relative_height_mm": 0.1},
            {"distance_mm": 0.3, "relative_height_mm": nan},
            {"distance_mm": 0.4, "relative_height_mm": 0.4},
            {"distance_mm": 0.5, "relative_height_mm": nan},
        ]
    ).set_index("distance_mm")
    trace = apply_dropout_correction(trace)
    assert np.array_equal(trace["relative_height_mm"], [0.1, 0.1, 0.3, 0.4, 0.4], True)


def test_apply_spike_removal_middle():

    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.1, 0.2, 0.3, 0.4],
            "relative_height_mm": [0.1, 0, 0.3, 0, 0.1],
        }
    ).set_index("distance_mm")
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(trace["relative_height_mm"], [0.1, 0.1, 0.1, 0.1, 0.1], True)


def test_apply_spike_removal_start_end():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.1, 0.2, 0.3, 0.4],
            "relative_height_mm": [0.3, 0, 0.1, 0, 0.3],
        }
    ).set_index("distance_mm")
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(trace["relative_height_mm"], [0.1, 0.1, 0.1, 0.1, 0.1], True)


def test_apply_spike_removal():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "relative_height_mm": [0.3, 0, 0.2, 0, 0.3, 0, -0.2, 0, 0.3],
        }
    ).set_index("distance_mm")
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(
        trace["relative_height_mm"].round(6),
        [0.2, 0.2, 0.2, 0.1, 0, -0.1, -0.2, -0.2, -0.2],
        equal_nan=True,
    )


def test_apply_spike_removal_no_spikes():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.1, 0.2, 0.3, 0.4],
            "relative_height_mm": [0, 0.1, 0.2, 0.1, -0.1],
        }
    ).set_index("distance_mm")
    trace = apply_spike_removal(trace, alpha=3)
    assert np.array_equal(trace["relative_height_mm"], [0, 0.1, 0.2, 0.1, -0.1], True)


def test_apply_lowpass_filter():
    trace = pd.DataFrame(
        {
            "relative_height_mm": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
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
        ],
    )


def test_build_resampled_trace():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
            "relative_height_mm": [
                0.3,
                0.1,
                0.2,
                0.2,
                0.1,
                0.2,
                0.1,
                -0.1,
                0.1,
                0.1,
                0.5,
            ],
        }
    ).set_index("distance_mm")
    resampled_trace = build_resampled_trace(trace, target_sample_spacing_mm=0.5)

    assert np.array_equal(resampled_trace.index, [0.5, 1, 1.5, 2, 2.5])
    assert np.array_equal(
        resampled_trace["relative_height_mm"], [0.2, 0.15, 0.15, 0, 0.3]
    )


def test_extract_segment_traces_from_trace():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "relative_height_mm": [
                0.3,
                0.1,
                0.2,
                0.2,
                0.1,
                0.2,
                0.1,
                -0.1,
                0.1,
                0.1,
                0.5,
            ],
        }
    ).set_index("distance_mm")
    segment_bins = [0, 100, 200, 300, 400, 500]
    segment_traces = list(extract_segment_traces_from_trace(trace, segment_bins))

    assert len(segment_traces) == 5
    assert [len(tt) for tt in segment_traces] == [3, 2, 2, 2, 2]
    assert np.array_equal(segment_traces[1].index, [150, 200])
    assert np.array_equal(segment_traces[1]["relative_height_mm"], [0.2, 0.1])


def test_extract_segment_data_segment_length():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
            "relative_height_mm": [
                0.3,
                0.1,
                0.2,
                0.2,
                0.1,
                0.2,
                0.1,
                -0.1,
                0.1,
                0.1,
                0.5,
            ],
        }
    ).set_index("distance_mm")
    resampled_trace = pd.DataFrame(
        {
            "distance_mm": [0.5, 1, 1.5, 2, 2.5],
            "relative_height_mm": [0.2, 0.15, 0.15, 0, 0.3],
        }
    ).set_index("distance_mm")
    trace_data = list(extract_segment_data(trace, resampled_trace, segment_length_mm=1))

    assert len(trace_data) == 3
    assert list(zip(*trace_data))[2] == (1, 1, 1)


def test_extract_segment_data_segment_bins():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
            "relative_height_mm": [
                0.3,
                0.1,
                0.2,
                0.2,
                0.1,
                0.2,
                0.1,
                -0.1,
                0.1,
                0.1,
                0.5,
            ],
        }
    ).set_index("distance_mm")
    resampled_trace = pd.DataFrame(
        {
            "distance_mm": [0.5, 1, 1.5, 2, 2.5],
            "relative_height_mm": [0.2, 0.15, 0.15, 0, 0.3],
        }
    ).set_index("distance_mm")
    trace_data = list(
        extract_segment_data(trace, resampled_trace, segment_bins=[0, 1, 2, 3])
    )

    assert len(trace_data) == 3
    assert list(zip(*trace_data))[2] == (1, 1, 1)


def test_apply_slope_correction():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
            "relative_height_mm": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    ).set_index("distance_mm")
    slope_suppressed_trace = apply_slope_correction(trace)
    assert np.array_equal(slope_suppressed_trace["relative_height_mm"], [0] * 11)


def test_calculate_msd():
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "relative_height_mm": [
                1,
                0,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                0,
            ],
        }
    ).set_index("distance_mm")
    assert calculate_msd(trace) == ((1 + 2) / 2) - (3 / 10)


def test_calculate_msd_divide_segment_false():
    """
    Test the divide segment option for the calculate_msd function. This option
    will calculate the mean square deviation of the trace without splitting
    the segment into two parts. I.e. The MSD is taken as the difference
    between the single largest peak height and the mean of the trace.
    """
    trace = pd.DataFrame(
        {
            "distance_mm": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "relative_height_mm": [
                1,
                0,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                0,
            ],
        }
    ).set_index("distance_mm")
    assert calculate_msd(trace, divide_segment=False) == 2.0 - (3.0 / 10)


def test_find_plates_start_only(data_path):
    path = data_path.joinpath("captif_profiles", "20211011_aylesbury_dec_station_0.dat")
    _, trace = load_reading(path)
    trace.set_index("distance_mm", inplace=True)
    trace = trace.iloc[: int(len(trace) / 2)]
    start_mm, end_mm = find_plates(trace)
    assert start_mm == 62.32
    assert end_mm is None


def test_find_plates_end_only(data_path):
    path = data_path.joinpath("captif_profiles", "20211011_aylesbury_dec_station_0.dat")
    _, trace = load_reading(path)
    trace.set_index("distance_mm", inplace=True)
    trace = trace.iloc[int(len(trace) / 2) :]
    start_mm, end_mm = find_plates(trace)
    assert start_mm is None
    assert end_mm == 1870.388


def test_find_plates(data_path):
    path = data_path.joinpath("captif_profiles", "20211011_aylesbury_dec_station_0.dat")
    _, trace = load_reading(path)
    trace.set_index("distance_mm", inplace=True)
    start_mm, end_mm = find_plates(trace)
    assert start_mm == 62.32

    assert end_mm == 1870.388


def test_find_plates_no_plates(data_path):
    path = data_path.joinpath("captif_profiles", "20211011_aylesbury_dec_station_0.dat")
    _, trace = load_reading(path)
    trace = trace.loc[
        (trace["distance_mm"] > 100) & (trace["distance_mm"] < 1800)
    ].reset_index(drop=True)
    trace.set_index("distance_mm", inplace=True)
    start_mm, end_mm = find_plates(trace)
    assert start_mm is None
    assert end_mm is None


class TestReading:
    def test_from_trace(self):
        trace = pd.DataFrame(
            {
                "distance_mm": np.arange(0, 10, 0.25),
                "relative_height_mm": [0.1] * 40,
            }
        ).set_index("distance_mm")
        reading = Reading.from_trace(trace)

        pd.testing.assert_frame_equal(reading.trace, trace, check_dtype=False)
        assert len(reading.resampled_trace) == 20

    def test_from_trace_using_segment_bins(self):
        trace = pd.DataFrame(
            {
                "distance_mm": np.arange(0, 10, 0.25),
                "relative_height_mm": [0.1] * 40,
            }
        ).set_index("distance_mm")
        reading = Reading.from_trace(trace, segment_bins=np.arange(0, 10, 1))

        assert reading.segment_length_mm is None
        assert np.array_equal(reading.segment_bins, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert len(reading.segments) == 9
        assert all((ss.segment_length_mm == 1 for ss in reading.segments))

    def test_from_file_4102a5dd(self):
        pass

    def test_segments(self):
        pass

    def test_msd_divide_segments_false(self):
        resampled_trace = pd.DataFrame(
            {  # using semetric trace to avoid slope correction
                "distance_mm": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                "relative_height_mm": [
                    1,
                    0,
                    0,
                    3,
                    0,
                    0,
                    3,
                    0,
                    0,
                    1,
                ],
                "dropout": [False] * 10,
                "spike": [False] * 10,
            }
        ).set_index("distance_mm")
        reading = Reading(
            meta={},
            trace=resampled_trace,
            resampled_trace=resampled_trace,
            resampled_sample_spacing_mm=10,
            alpha=3,
            segment_length_mm=100,
            divide_segments=False,
        )
        assert reading.msd() == [
            {
                "segment_no": 0,
                "msd": 3.0 - (8.0 / 10),
                "valid": True,
                "evaluation_length_position_m": None,
            }
        ]


class TestSegment:
    def test_dropout_ratio(self):
        trace = pd.DataFrame({"dropout": [True, False, True, False, False]})
        segment = Segment(segment_no=1, trace=trace, resampled_trace=pd.DataFrame())
        assert segment.dropout_ratio == 2 / 5

    def test_spike_ratio(self):
        resampled_trace = pd.DataFrame({"spike": [False, False, True, False, False]})
        segment = Segment(
            segment_no=1, trace=pd.DataFrame(), resampled_trace=resampled_trace
        )
        assert segment.spike_ratio == 1 / 5

    def test_is_valid(self):
        assert (
            Segment(
                segment_no=1,
                trace=pd.DataFrame({"dropout": [True] * 1 + [False] * 9}),
                resampled_trace=pd.DataFrame({"spike": [False] * 10}),
            ).is_valid
            is True
        )

        assert (
            Segment(
                segment_no=1,
                trace=pd.DataFrame({"dropout": [True] * 2 + [False] * 8}),
                resampled_trace=pd.DataFrame({"spike": [False] * 10}),
            ).is_valid
            is False
        )

    def test_is_valid_custom_allowed_dropout_percent(self):
        assert (
            Segment(
                segment_no=1,
                trace=pd.DataFrame({"dropout": [True] * 2 + [False] * 8}),
                resampled_trace=pd.DataFrame({"spike": [False] * 10}),
                allowed_dropout_percent=20,
            ).is_valid
            is True
        )

        assert (
            Segment(
                segment_no=1,
                trace=pd.DataFrame({"dropout": [True] * 3 + [False] * 7}),
                resampled_trace=pd.DataFrame({"spike": [False] * 10}),
                allowed_dropout_percent=20,
            ).is_valid
            is False
        )

    def test_msd_divide_segment_false(self):
        resampled_trace = pd.DataFrame(
            {
                "distance_mm": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                "relative_height_mm": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        ).set_index("distance_mm")
        segment = Segment(
            segment_no=1,
            trace=pd.DataFrame(),
            resampled_trace=resampled_trace,
            divide_segment=False,
        )
        assert segment.msd == 3.0 - (4.0 / 10)
