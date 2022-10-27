from datetime import datetime
from pydantic import BaseModel, validator, ValidationError
from pyrsona import BaseStructure
from typing import Optional, Union


class FileStructure(BaseStructure):
    pass


class CaptifSlpFileStructure(FileStructure):
    class meta_model(BaseModel):

        datetime: Optional[datetime]
        file_number: Optional[int]

        @validator("datetime", pre=True)
        def parse_date(cls, value):
            fmt = ["%d/%m/%Y\t%I:%M %p", "%d/%m/%Y"]
            for ff in fmt:
                try:
                    return datetime.strptime(value, ff)
                except Exception:
                    pass
            raise ValidationError

    class row_model(BaseModel):

        point_no: Optional[Union[int, float]]
        distance_mm: float
        relative_height_mm: Optional[float]

        @validator("relative_height_mm", pre=True)
        def parse_relative_height_mm(cls, value):
            return None if value == "NaN" else value

    @staticmethod
    def table_postprocessor(table_rows, meta):
        for ii, _ in enumerate(table_rows):
            del table_rows[ii]["point_no"]
        return table_rows


class _4102a5dd(CaptifSlpFileStructure):

    structure = (
        "Road Name\tCAPTIF\n"
        "Ref Station\t{}\n"
        "Start Pos (m)\t{}\n"
        "End Pos (m)\t{}\n"
        "Direction\t{}\n"
        "Wheel Path\t{}\n"
        "Date\t{datetime}\n"
        "File No.\t{file_number}\n"
        "Current Pos\t{}\n"
        "*****DATA*****\t\n"
    )


class _245ff223(CaptifSlpFileStructure):

    structure = (
        "Road Name\tCAPTIF\t\n"
        "Ref Station\t{}\t\n"
        "Start Pos (m)\t{}\t\n"
        "End Pos (m)\t{}\t\n"
        "Direction\t{}\t\n"
        "Wheel Path\t{}\t\n"
        "Date\t{datetime}\t\n"
        "File No.\t{file_number}\t\n"
        "Current Pos\t{}\t\n"
        "*****DATA*****\t\t\n"
        "Data Point No:\tDistance (mm)\tDepth (mm)\n"
    )


class _0319aee1(CaptifSlpFileStructure):

    structure = (
        "Road Name\tCAPTIF\t\n"
        "Ref Station\t\t\n"
        "Start Pos (m)\t{}\t\n"
        "Direction\t{}\t\n"
        "Wheel Path\t{}\t\n"
        "Date\t{datetime}\n"
        "File No.\t{file_number}\t\n"
        "Current Pos\t{}\t\n"
        "*****DATA*****\t\t\n"
        "Data Point No:\tDistance (mm)\tDepth (mm)\n"
    )


class _c5084427(CaptifSlpFileStructure):

    structure = (
        "Road Name\t{}\t\n"
        "Ref Station\t{}\t\n"
        "Start Pos (m)\t{}\t\n"
        "Direction\t{}\t\n"
        "Wheel Path\t{}\t\n"
        "Date\t{datetime}\n"
        "File No.\t{file_number}\t\n"
        "Current Pos\t{}\t\n"
        "*****DATA*****\t\t\n"
        "Data Point No:\tDistance (mm)\tDepth (mm)\n"
    )


class _a2348fea(CaptifSlpFileStructure):

    structure = (
        "Road Name\t{}\t\n"
        "Ref Station\t\t\n"
        "Start Pos (m)\t{}\t\n"
        "Direction\t{}\t\n"
        "Wheel Path\t{}\t\n"
        "Date\t{datetime}\n"
        "File No.\t{file_number}\t\n"
        "Current Pos\t{}\t\n"
        "*****DATA*****\t\t\n"
        "Data Point No:\tDistance (mm)\tDepth (mm)\n"
    )


class ErpugFileStructure(BaseStructure):
    class meta_model(BaseModel):

        sample_spacing_mm: float

    class row_model(BaseModel):

        relative_height_mm: Optional[float]

        @validator("relative_height_mm", pre=True)
        def parse_relative_height_mm(cls, value):
            return None if value == "NaN" else value

    @staticmethod
    def table_postprocessor(table_rows, meta):
        for ii, _ in enumerate(table_rows):
            table_rows[ii]["distance_mm"] = ii * meta.get("sample_spacing_mm")
        return table_rows


class _7cd12dee(ErpugFileStructure):

    structure = "profile_name: {}\n" "sample_spacing_mm: {sample_spacing_mm}\n"
