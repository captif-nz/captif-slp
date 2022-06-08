import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from functools import lru_cache


LOWPASS_FILTER_CUTOFF_WAVELENGTH_MM = 2.4
HIGHPASS_FILTER_CUTOFF_WAVELENGTH_MM = 174.2


@lru_cache()
def build_lowpass_filter(sample_spacing_mm: float, output="sos"):
    """
    Build a 2nd order low-pass butterworth filter as defined in Annex D of ISO 13473:2019.

    :param sample_spacing_mm: sampling interval (mm). Typically 0.5 or 1.0 mm.

    :return array of filter coefficients.

    """
    highcut = 1 / LOWPASS_FILTER_CUTOFF_WAVELENGTH_MM
    fs = 1 / sample_spacing_mm
    nyq = 0.5 * fs
    high = highcut / nyq
    return signal.butter(2, high, btype="low", output=output)


@lru_cache()
def build_highpass_filter(sample_spacing_mm: float, output="sos"):
    """
    Build a 2nd order high-pass butterworth filter as defined in Annex D of
    ISO 13473:2019.

    :param sample_spacing_mm: sampling interval (mm). Typically 0.5 or 1.0 mm.

    :return array of filter coefficients.

    """
    lowcut = 1 / HIGHPASS_FILTER_CUTOFF_WAVELENGTH_MM
    fs = 1 / sample_spacing_mm
    nyq = 0.5 * fs
    low = lowcut / nyq
    return signal.butter(2, low, btype="high", output=output)


def plot_filter_response(sos: np.array, fs: float):
    w, h = signal.sosfreqz(sos, worN=2000)

    fig, ax = plt.subplots()
    ax.plot((fs * 0.5 / np.pi) * w, abs(h))
    ax.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], "--", label="sqrt(0.5)")

    ax.set_xlabel("Frequency (mm-1)")
    ax.set_ylabel("Gain")
    ax.grid(True)
    ax.legend(loc="best")

    return fig, ax


def generate_ba_filter_coefficients(
    filter_type,
    sample_spacing_mm,
    normalise=True,
):
    if filter_type == "lowpass":
        b, a = build_lowpass_filter(sample_spacing_mm, "ba")
    if filter_type == "highpass":
        b, a = build_highpass_filter(sample_spacing_mm, "ba")

    if normalise:
        a = a / b[0]
        b = b / b[0]

    return b, a


def generate_reference_filter_coefficients(normalise=True):
    filter_coefficients = []
    for filter_type in ["lowpass", "highpass"]:
        for sample_spacing_mm in [0.5, 1]:
            b, a = generate_ba_filter_coefficients(
                filter_type,
                sample_spacing_mm,
                normalise,
            )
            filter_coefficients.append(
                {
                    "filter_type": filter_type,
                    "sample_spacing_mm": sample_spacing_mm,
                    "b": b.tolist(),
                    "a": a.tolist(),
                }
            )
    return filter_coefficients
