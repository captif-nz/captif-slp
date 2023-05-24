from captif_slp.process import process_files


def test_process_files(data_path):
    path = data_path.joinpath("captif_profiles")
    results = process_files(path)

    assert len(results) == 3  # expect results from 3 files
    assert all(
        (len(rr.keys()) == 11 for rr in results)
    )  # expect 11 result fields per file
    assert all((len(rr["trace"]) > 0 for rr in results))  # expect trace field

    # Check MPD and standard deviation are correct:
    results_ = [
        {"mpd": 3.420159709285714, "stdev": 0.8103657268479817},
        {"mpd": 3.0331992396875003, "stdev": 0.44966725032778604},
        {"mpd": 3.2777511138235296, "stdev": 0.8530880184893223},
    ]
    assert all(
        (rr[kk] == vv for rr, rr_ in zip(results, results_) for kk, vv in rr_.items())
    )
