import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer

from utils import create_classification_targets, train_classifier


def main(argv=None):
    parser = argparse.ArgumentParser("Evaluating BERT integer embeddings")

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
    args = parser.parse_args(argv)
    model_name = "bert-base-uncased"

    # Create writer
    writer = SummaryWriter(args.log_folder)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Retrieve embeddings
    to_find = list(map(str, range(args.max_value_eval)))
    positions = np.array(tokenizer.convert_tokens_to_ids(to_find))
    unk_token_position = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    is_valid = positions != unk_token_position

    print(
        "The following numbers are missing",
        [i for i, x in enumerate(is_valid) if not x],
    )

    arange = np.arange(args.max_value_eval)
    numbers = arange[is_valid]
    embeddings = (
        model.embeddings.word_embeddings(torch.from_numpy(positions[is_valid]))
        .detach()
        .numpy()
    )

    ys_clf = create_classification_targets(numbers)

    keys = sorted(ys_clf.keys())
    metadata = np.array([numbers] + [ys_clf[k] for k in keys]).T.tolist()
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
