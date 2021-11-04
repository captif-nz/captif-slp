
import numpy as np
from scipy.signal import sosfreqz

from captif_slp import signal


def test_build_lowpass_filter_check_coefficients():
    sos = signal.build_lowpass_filter(0.5)
    assert np.array_equal(
        sos[0],
        [
            0.22019470027295873,
            0.44038940054591746,
            0.22019470027295873,
            1.0,
            -0.30756635979220975,
            0.18834516088404465,
        ]
    )


def test_build_lowpass_filter_sos_3db_down_point():
    sos = signal.build_lowpass_filter(0.5)
    freq, h = sosfreqz(sos, fs=1/0.5)
    magnitude = abs(h)

    highcut_wavelength = 1 / freq[magnitude <= np.sqrt(0.5)][0]

    assert abs(1 - (highcut_wavelength / 2.4)) <= 0.005  # check within 0.5% of target

