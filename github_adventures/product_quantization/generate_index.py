from __future__ import annotations

import argparse
import logging
import pathlib
import pickle

import faiss
import numpy as np

from custom import CustomIndexPQ

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_path",
    type=pathlib.Path,
    help="Path to the full embeddings array",
)
parser.add_argument(
    "index_type",
    type=str,
    choices=["faiss-flat", "faiss-pq", "our-pq"],
    help="Type of index to generate",
)
parser.add_argument(
    "output_path",
    type=pathlib.Path,
    help="Path to where to store the index"
)

args, unknown_kwargs = parser.parse_known_args()
hyperparams: dict[str, int] = {}

for i in range(0, len(unknown_kwargs), 2):
    key_raw, value_raw = unknown_kwargs[i], unknown_kwargs[i + 1]

    key = key_raw.strip("--")
    value = int(value_raw) if value_raw.isnumeric() else value_raw
    hyperparams[key] = value

logger.info(f"The following hyperparameters were detected {hyperparams}")
logger.info("Loading embeddings")
embs = np.load(args.input_path)
n, d = embs.shape

if args.index_type == "faiss-flat":
    logger.info("Instantiating IndexFlatL2")
    index = faiss.IndexFlatL2(d)

elif args.index_type == "faiss-pq":
    logger.info("Instantiating IndexPQ")
    arguments = [d, hyperparams["m"], hyperparams["nbits"]]
    index = faiss.IndexPQ(*arguments)

elif args.index_type == "our-pq":
    logger.info("Instantiating CustomIndexPQ")
    index = CustomIndexPQ(d, **hyperparams)

logger.info("Training the index")
index.train(embs)

logger.info("Adding all embeddings to the index")
index.add(embs)

logger.info(f"Writing index to disk - {args.output_path}")

if args.index_type == "our-pq":
    with args.output_path.open("wb") as f:
        pickle.dump(index, f)
    
else:
    faiss.write_index(index, str(args.output_path))
