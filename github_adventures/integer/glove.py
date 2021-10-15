import argparse

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import create_classification_targets, train_classifier


def main(argv=None):
    parser = argparse.ArgumentParser("Evaluating GloVe integer embeddings")

    parser.add_argument(
        "glove_path",
        type=str,
        help="Path to a txt file holding the GloVe embeddings",
    )
    parser.add_argument(
        "log_folder",
        type=str,
        help="Folder where to log results",
    )
    parser.add_argument(
        "--max-value-eval",
        type=int,
        default=500,
        help="Number of integers to run the evaluation on",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=300,
        help="Dimensionality of the embeddings",
    )
    args = parser.parse_args()

    # Create writer
    writer = SummaryWriter(args.log_folder)

    # Retrieve embeddings
    to_find = set(map(str, range(args.max_value_eval)))
    embeddings = np.empty((args.max_value_eval, args.dim))

    with open(args.glove_path) as f:
        for line in f:
            token, *vector_ = line.split(" ")

            if token in to_find:
                embeddings[int(token)] = list(map(float, vector_))
                to_find.remove(token)

    assert not to_find

    arange = np.arange(args.max_value_eval)
    ys_clf = create_classification_targets(arange)

    keys = sorted(ys_clf.keys())
    metadata = np.array([arange] + [ys_clf[k] for k in keys]).T.tolist()
    metadata_header = ["value"] + keys

    for name, y in ys_clf.items():
        metrics = train_classifier(embeddings, y)
        for metric_name, value in metrics.items():
            writer.add_scalar(
                f"{name}/{metric_name}",
                value,
            )

    writer.add_embedding(
        embeddings,
        metadata=metadata,
        metadata_header=metadata_header,
    )


if __name__ == "__main__":
    main()
