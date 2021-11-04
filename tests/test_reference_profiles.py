
import pytest

from captif_slp import Reading


@pytest.mark.slow
class TestReferenceProfiles:
    """
    Calculate the mean profile depth for the eight ERPUG reference profiles (see
    https://www.erpug.org/index.php?contentID=239) and compare the results to the VTI
    results.

    The tests will pass if the mean MPD value for each profile is within 0.1% of the VTI
    value.

    """

    def test_profile1(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj01.mm")
        mpd_target = 2.486563

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001

    def test_profile2(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj02.mm")
        mpd_target = 0.739969

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001

    def test_profile3(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj03.mm")
        mpd_target = 1.688548

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.0015

    def test_profile4(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj04.mm")
        mpd_target = 1.178962

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001

    def test_profile5(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj05.mm")
        mpd_target = 0.649049

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001

    def test_profile6(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj06.mm")
        mpd_target = 0.417096

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001

    def test_profile7(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj07.mm")
        mpd_target = 1.314730

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001

    def test_profile8_long_target_sample_spacing(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj08.mm")
        mpd_target = 0.806073

        reading = Reading.from_file(
            path, target_sample_spacing_mm=1.0, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001

    def test_profile8_short_target_sample_spacing(self, data_path):
        path = data_path.joinpath("erpug_profiles", "obj08.mm")
        mpd_target = 0.818080

        reading = Reading.from_file(
            path, target_sample_spacing_mm=0.5, evaluation_length_m=20,
        )
        results = reading.mpd()
        assert abs(1 - (results["mean"].mean() / mpd_target)) <= 0.001
