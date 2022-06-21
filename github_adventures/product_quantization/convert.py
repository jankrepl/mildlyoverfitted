import argparse
import logging
import pathlib
import pickle

import faiss

from custom import CustomIndexPQ

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def from_faiss(faiss_index: faiss.swigfaiss.IndexPQ) -> CustomIndexPQ:
    if not faiss_index.is_trained:
        raise ValueError("The faiss index is not trained")

    if faiss_index.ntotal == 0:
        raise ValueError("The faiss index has no codes")

    d = faiss_index.d
    m = faiss_index.code_size
    nbits = faiss_index.pq.nbits
    k = 2**nbits
    ntotal = faiss_index.ntotal

    custom_index = CustomIndexPQ(d=d, m=m, nbits=nbits)
    centers = faiss.vector_to_array(faiss_index.pq.centroids).reshape(
        m, k, d // m
    )

    logger.info("Copying centers from the faiss index")
    for i in range(m):
        custom_index.estimators[i].cluster_centers_ = centers[i]
    custom_index.is_trained = True

    logger.info("Copying codes form the faiss index")
    custom_index.codes = faiss.vector_to_array(faiss_index.codes).reshape(
        ntotal, m
    )

    return custom_index


def main() -> int:
    parser = argparse.ArgumentParser("Convert from faiss to custom")
    parser.add_argument(
        "faiss_index_path",
        type=pathlib.Path,
        help="Path to a faiss index",
    )
    parser.add_argument(
        "output_index_path",
        type=pathlib.Path,
        help="Path to a new custom index with faiss parameters",
    )

    args = parser.parse_args()

    faiss_index = faiss.read_index(str(args.faiss_index_path))
    custom_index = from_faiss(faiss_index)

    with args.output_index_path.open("wb") as f:
        pickle.dump(custom_index, f)


if __name__ == "__main__":
    main()
