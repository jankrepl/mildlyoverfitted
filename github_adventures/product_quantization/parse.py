from __future__ import annotations

import argparse
import io
import logging
import pathlib
import tqdm

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_embeddings(path: str, maximum: int | None = None) -> tuple[list[str], np.ndarray]:
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    n = n if maximum is None else min(n, maximum)

    embs: np.ndarray = np.empty((n, d), dtype=np.float32)
    words: list[str] = []

    for i, line in tqdm.tqdm(enumerate(fin)):
        if maximum is not None and i == maximum:
            break

        tokens = line.rstrip().split(' ')

        words.append(tokens[0])
        embs[i] = list(map(float, tokens[1:]))
    
    return words, embs

parser = argparse.ArgumentParser()
parser.add_argument(
    "fasttext_path",
    type=pathlib.Path,
    help="Path to fasttext embeddings.",
)
parser.add_argument(
    "output_dir",
    type=pathlib.Path,
    help="Directory where we store the words and the embeddings."
)
parser.add_argument(
    "-m",
    "--max",
    type=int,
    help="Maximum number of embeddings to parse."
)

args = parser.parse_args()

path_embs = args.output_dir / "embs.npy"
path_words = args.output_dir / "words.txt"

args.output_dir.mkdir(exist_ok=True, parents=True)

logger.info("Parsing")
words, embs = get_embeddings(args.fasttext_path, maximum=args.max)

logger.info("Saving words")
with path_words.open("w") as f:
    for word in words:
        f.write(word + "\n")
    
logger.info("Saving embeddings")
np.save(path_embs, embs)

