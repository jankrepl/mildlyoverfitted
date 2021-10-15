import pathlib
import pickle

import requests

from joblib import Parallel, delayed, parallel_backend


def get_sequence(sequence_id):
    """Get an integer sequence from the online OEIS.

    Parameters
    ----------
    sequence_id : int
        Unique identifier for the desired sequence.

    Returns
    -------
    sequence : list
        List of integers

    Raises
    ------
    HTTPError
        Was not possible to get the given sequence
    """
    url = f"https://oeis.org/search?fmt=json&q=id:A{sequence_id:07}"
    print(sequence_id)
    response = requests.get(url)

    response.raise_for_status()

    data_str = response.json()["results"][0]["data"]
    sequence = [int(x) for x in data_str.split(",")]

    return sequence


if __name__ == "__main__":
    # Parameters
    n_sequences = 5000
    start_id = 1  # seems like 1 - 340_000 are valid sequences
    n_jobs = 64
    backend = "threading"  # "threading" or "loky"

    # Preparation
    end_id = start_id + n_sequences
    output_folder = pathlib.Path("data/")
    output_folder.mkdir(exist_ok=True, parents=True)
    output_path = output_folder / f"{start_id}_{end_id - 1}.pkl"

    with parallel_backend(backend, n_jobs=n_jobs):
        res = Parallel()(delayed(get_sequence)(i) for i in range(start_id, end_id))

    with output_path.open("wb") as f:
        pickle.dump(res, f)

