from datetime import datetime

from captif_slp.structures import CaptifSlpFileStructure, ErpugFileStructure


class TestCaptifSlpFileStructure:
    def test_captif_slp_file_structure_4102a5dd(self, data_path):
        path = data_path.joinpath("structures", "4102a5dd.dat")
        meta, table_rows, structure_id = CaptifSlpFileStructure.read(path)

        assert meta == {
            "datetime": datetime(2019, 1, 16),
            "file_number": 0,
        }
        assert table_rows == [
            {"distance_mm": 0, "relative_height_mm": None},
            {"distance_mm": 0.037, "relative_height_mm": None},
            {"distance_mm": 0.075, "relative_height_mm": None},
            {"distance_mm": 0.112, "relative_height_mm": None},
            {"distance_mm": 0.150, "relative_height_mm": None},
            {"distance_mm": 0.187, "relative_height_mm": -0.122},
            {"distance_mm": 0.225, "relative_height_mm": -0.065},
        ]
        assert structure_id == "4102a5dd"

    def test_texture_reader_245ff223(self, data_path):
        path = data_path.joinpath("structures", "245ff223.dat")
        meta, table_rows, structure_id = CaptifSlpFileStructure.read(path)

        assert meta == {
            "datetime": datetime(2021, 9, 29),
            "file_number": 1,
        }
        assert table_rows == [
            {"distance_mm": 0, "relative_height_mm": -2.232},
            {"distance_mm": 0.037, "relative_height_mm": None},
            {"distance_mm": 0.075, "relative_height_mm": -2.090},
        ]
        assert structure_id == "245ff223"

    def test_texture_reader_0319aee1(self, data_path):
        path = data_path.joinpath("structures", "0319aee1.dat")
        meta, table_rows, structure_id = CaptifSlpFileStructure.read(path)

        assert meta == {
            "datetime": datetime(2021, 10, 22, 8, 16),
            "file_number": 0,
        }
        assert table_rows == [
            {"distance_mm": 0, "relative_height_mm": -0.027},
            {"distance_mm": 0.037, "relative_height_mm": -0.044},
            {"distance_mm": 0.075, "relative_height_mm": 0.019},
        ]
        assert structure_id == "0319aee1"

    def test_texture_reader_c5084427(self, data_path):
        path = data_path.joinpath("structures", "c5084427.dat")
        meta, table_rows, structure_id = CaptifSlpFileStructure.read(path)

        assert meta == {
            "datetime": datetime(2022, 6, 8, 10, 50),
            "file_number": 0,
        }
        assert table_rows == [
            {"distance_mm": 0, "relative_height_mm": 2.538},
            {"distance_mm": 0.037, "relative_height_mm": 2.554},
            {"distance_mm": 0.075, "relative_height_mm": 2.535},
        ]
        assert structure_id == "c5084427"

    def test_texture_reader_a2348fea(self, data_path):
        path = data_path.joinpath("structures", "a2348fea.dat")
        meta, table_rows, structure_id = CaptifSlpFileStructure.read(path)

        assert meta == {
            "datetime": datetime(2022, 3, 4, 0, 20, 0),
            "file_number": 54,
        }
        assert table_rows == [
            {"distance_mm": 0, "relative_height_mm": 1.032},
            {"distance_mm": 0.037, "relative_height_mm": 0.975},
            {"distance_mm": 0.075, "relative_height_mm": 0.902},
            {"distance_mm": 0.112, "relative_height_mm": 0.878},
        ]
        assert structure_id == "a2348fea"


class TestErpugFileStructure:
    def test_texture_reader_7cd12dee(self, data_path):
        path = data_path.joinpath("structures", "7cd12dee.dat")
        meta, table_rows, structure_id = ErpugFileStructure.read(path)

        assert meta == {"sample_spacing_mm": 1.0}
        assert table_rows == [
            {"distance_mm": 0, "relative_height_mm": -7.406126},
            {"distance_mm": 1.0, "relative_height_mm": -7.214107},
            {"distance_mm": 2.0, "relative_height_mm": -7.239000},
            {"distance_mm": 3.0, "relative_height_mm": -7.295854},
        ]
        assert structure_id == "7cd12dee"
