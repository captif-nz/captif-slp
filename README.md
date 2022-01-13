# captif-slp

The *captif-slp* Python package can be used to calculate the Mean Profile Depth (MPD) of a road surface according to *[ISO 13473-1:2019](https://www.iso.org/standard/45111.html) Characterization of pavement texture by use of surface profiles — Part 1: Determination of mean profile depth*.

The software has been developed at the Waka Kotahi NZ Transport Agency *[CAPTIF Road Research Centre](https://www.nzta.govt.nz/roads-and-rail/road-composition/road-pavements/captif/)* for use with the CAPTIF in-house Stationary Laser Profilometer (SLP). While some methods are specific to data captured using the CAPTIF SLP instrument the software can also be used to calculate MPD of other road surface profiles provided the data is converted to a valid format before processing.

Refer to the *[European Road Profile Users' Group](https://www.erpug.org/index.php?contentID=239)* website for further information on ISO 13473-1 including a MATLAB implementation and a set of reference profiles and results.

Also refer to *[Transfund New Zealand Research Report 84](https://www.nzta.govt.nz/resources/research/reports/84/)* for details of the original Stationary Laser Profilometer developed in 1997 for use by Transit New Zealand. This instrument was based on the VTI stationary laser profilometer.

## Installation

*captif-slp* can be installed by running `pip install captif-slp`.

The package requires Python 3.9 to run.

## Usage

### The Reading object

The *Reading* class provides a mechanism to interact with an individual road surface profile and can be imported using `from captif_slp import Reading`.

A new *Reading* object must be created using one of the following class methods.

1. From a DataFrame containing `distance_mm` and `relative_height_mm` columns:

    ```python
    reading = Reading.from_trace(
        trace: DataFrame,
        meta: Optional[dict] = None,
        ...
    )
    ```

    The `meta` keyword argument provides a mechanism for attaching meta data to a *Reading* object. This can be used to include information such as a file name, reading number or measurement date. The data can be accessed later from the *meta* attribute.

2. Or from a valid *texture* data file capable of being read using the *[captif-data-structures](https://github.com/captif-nz/captif-data-structures)* package (i.e. a CAPTIF in-house file format):

    ```python
    reading = Reading.from_file(
        path: Union[str, Path],
        parallel: bool = True,
        ...
    )
    ```

    The `parallel` keyword argument allows the file to be read using the Python *multiprocessing* module. This can significantly speed up load times for large files. Note that where several files are being processed it is more efficient to pass individual files to each CPU core for processing in which case *parallel* **must be set to False** to allow the multiprocessing module to be used by a higher level function.

The remaining keyword arguments are common for `Reading.from_trace()` and `Reading.from_file()`:

```python
reading = Reading.from_...(
    ...
    segment_length_mm: int = 100,
    target_sample_spacing_mm: float = 0.5,
    evaluation_length_m: Optional[float] = None,
    alpha: int = 3,
    start_mm: Optional[float] = None,
    end_mm: Optional[float] = None,
    detect_plates: bool = False,
)
```

The keyword arguments have the following meanings:

- `segment_lenght_mm` -- *default: 100*
\
Length of the segments (in millimetres) that the profile is split into for processing. Note that this should generally be left as the default value of 100 mm as this is the required segment length in ISO 13473-1.

- `target_sample_spacing_mm` -- *default: 0.5*
\
The sample spacing (in millimetres) used by the resampling process. ISO 13473-1:2019 calls for a road surface profile to be resampled to either 0.5 mm or 1.0 mm depending on the sample spacing in the original profile. In general 0.5 mm should be used where the original sample spacing is less than or equal to 0.5 mm.

- `evaluation_length_m` -- *default: None*
\
The evaluation length (in metres) over which the mean profile depth is reported. The default behaviour is to calculate MPD for the entire road surface profile. For spot measurements taken using a Stationary Laser Profilometer (SLP) the default behaviour is appropriate. For continuous measurements it is usually more useful to report MPD over shorter sections in which case the evaluation length could be set to 10 or 20 metres.

- `alpha` -- *default: 3*
\
The constant used in the spike detection criterion (Eqn E.1, ISO 13473-1:2019). The default value of 3 is the value suggested in the standard. Note that a value of 3 used with profiles taken using the CAPTIF SLP results in a large number of false positives when analysing highly texture surfaces in New Zealand (e.g. chipseals). A value of 6 has been found to yield good results on profiles taken using the CAPTIF SLP instrument.

- `start_mm` -- *default: None*
\
Used to remove a section from the beginning of the profile before processing. The default is use the full profile. The value is ignored if the *detect_plates* is set to True.

- `end_mm` -- *default: None*
\
Used to remove a section from the end of the profile before processing. The default is use the full profile. The value is ignored if the *detect_plates* is set to True.

- `detect_plates` -- *default: False*
\
Used to detect reference plates at the start and/or end of the profile and trim the profile accordingly. For historical reasons reference plates are present on the CAPTIF SLP instrument to signal the start and end of the section of interest; these reference plates (along with the sections of profile outside of them) need to be removed from the profile before processing. The default behaviour is to not preform this step as it is unique to the CAPTIF SLP instrument.

#### Accessing the results

Once a profile has been successfully loaded into a *Reading* object the following instance methods can be used to access the results.

`reading.msd()` -- Mean Segment Depth (MSD)
\
Returns a list of dictionaries containing MSD values for each segment (usually 100 mm long) in the profile. Each *dict* element in the list has the following keys:

- `segment_no: int` - segment position number (int).
- `msd: float` - mean segment depth.
- `valid: bool` - segment validity (i.e. meets the maximum dropouts and spikes requirement).
- `evaluation_length_position_m: Optional[float]` - evaluation length position.

`reading.mpd(include_meta: bool = False)` -- Mean Profile Depth (MPD)
\
Returns either a single dictionary if evaluation length is not set, or a list of dictionaries if evaluation length is set. The *dict* element(s) have the following keys:

- `distance_m` - evaluation length position (*None* if evaluation length is not set).
- `mpd` - mean profile depth (average MSD across all valid segments).
- `stdev` - standard deviation (standard deviation of MSD from all valid segments).
- `valid_segments` - number of valid segments in the profile.
- `proportion_valid_segments` - proportion of valid segments in the profile.
- `is_valid` - result validity (i.e. whether or not valid_segment_ration >= 0.5).
- (meta data fields) - the meta data fields if *include_meta* is set to True.

### Bulk processing

In order to provide efficient processing the Python *multiprocessing* module is used to process batches of valid data files meeting the CAPTIF in-house *texture* file format. The processing function can be imported using `from captif_slp import process_files`.

The function takes the path of a folder containing the files to be processed and returns a list of results as per `Reading.mpd(include_meta=True)` with the addition of a `filename` and `folder` field.

```python
results = process_files(path: Union[str, Path])
```

`process_files` uses the following settings under the hood as these are the settings relevant to profiles taken using the CAPTIF SLP instrument:

- `segment_lenght_mm = 100`
- `target_sample_spacing_mm = 0.5`
- `evaluation_length_m = None`
- `alpha = 6`
- `detect_plates = True`

To access all attributes from `Reading.from_file` (except the *parallel* attribute which is set to False internally to allow the *multiprocessing* module to be used by a higher level) use `captif_slp.process.process_generic_files`.

## Issues

Please submit an issue if you find a bug or have an idea for an improvement.

## License

The software is provided under the following license.

*MIT License*

*Copyright (c) 2021 CAPTIF Road Research Centre*

*Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:*

*The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.*

*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.*
