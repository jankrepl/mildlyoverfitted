from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import time
from functools import partial
from typing import Any

import faiss
import gradio as gr
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "exact_index_path",
    type=pathlib.Path,
    help="Path to the exact index",
)
parser.add_argument(
    "approximate_index_path",
    type=pathlib.Path,
    nargs="+",
    help="Path to the approximate index",
)
parser.add_argument(
    "words_path",
    type=pathlib.Path,
    help="Path to the text file containing words",
)

args = parser.parse_args()


def run(
        word: str,
        k: int,
        exact_index,
        approximate_indexes: dict[str, Any],
        words: list[str],
        word2ix: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    metrics = {}

    emb = exact_index.reconstruct(word2ix[word])

    start = time.monotonic()
    D, I = exact_index.search(emb[None, :], k)
    metrics["time_exact"] = time.monotonic() - start
    D, I = D[0], I[0]

    df_e = pd.DataFrame({
        "ix": I,
        "distance": D,
        "word": [words[i] for i in I],
    })
    dfs_a = []

    for name, approximate_index in approximate_indexes.items():
        start = time.monotonic()
        D, I = approximate_index.search(emb[None, :], k)
        metrics[f"time_approximate_{name}"] = time.monotonic() - start
        D, I = D[0], I[0]

        df_a = pd.DataFrame({
            "ix": I,
            "distance": D,
            "word": [words[i] for i in I],
        })
        dfs_a.append(df_a)

        metrics[f"recall_{name}"] = len(np.intersect1d(df_e.word.unique(), df_a.word.unique())) / k

    return df_e, *dfs_a, metrics


logger.info(f"Loading words {args.words_path}")
words = args.words_path.read_text().strip().split("\n")
word2ix = {word: i for i, word in enumerate(words)}

logger.info(f"Loading exact index {args.exact_index_path}")
exact_index = faiss.read_index(str(args.exact_index_path))

logger.info(f"Loading approximate indexes {args.approximate_index_path}")

approximate_indexes = {
}

for path in args.approximate_index_path:
    if path.suffix in {".pkl", "pickle"}:
        with path.open("rb") as f:
            approximate_indexes[path.stem] = pickle.load(f)

    else:
        approximate_indexes[path.stem] = faiss.read_index(str(path))

# Sanity checks
assert isinstance(exact_index, faiss.IndexFlat)
# assert len(words) == exact_index.ntotal == approximate_index.ntotal

run_partial = partial(
    run,
    exact_index=exact_index,
    approximate_indexes=approximate_indexes,
    words=words,
    word2ix=word2ix,
)

setattr(run_partial, "__name__", "run_function")

demo = gr.Interface(
    fn=run_partial,
    inputs=[
        gr.Textbox(lines=1, placeholder="Word here..."),
        gr.Slider(minimum=1, maximum=20, value=5, step=1),
    ],
    outputs=[
        gr.DataFrame(label="exact"),
        *[gr.DataFrame(label=name) for name in approximate_indexes.keys()],
        gr.JSON(label="metrics"),
    ],
    allow_flagging="never",

)

demo.launch()
