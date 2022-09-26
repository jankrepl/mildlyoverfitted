import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sympy.ntheory import isprime
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Dataset containing integer sequences.

    Parameters
    ----------
    raw_sequences : list of list of str
        Containing the original raw sequences. Note
        that their length differs.

    sequence_len : int
        The lenght og the sequence. If the original sequence is shorter,
        we just pad it with `max_value`. If the original sequence is longer
        we simply cut if off.

    max_value : int
        The maximum allowed value (non inclusive). We will only consider
        sequences that had the first `sequence_len` elements in
        the range `[0, max_value)`.

    Attributes
    ----------
    normalized_sequences : np.ndarray
        2D array of shape `(n_sequences, sequence_len)`. It only contains
        sequences that had the first `sequence_len` elements in
        the range `[0, max_value)`.
    """

    def __init__(
            self,
            raw_sequences,
            sequence_len=80,
            max_value=2000,
    ):
        filtered_sequences = list(
            filter(
                lambda seq: all(
                    0 <= x < max_value for x in seq[:sequence_len]
                ),
                raw_sequences,
            )
        )

        n_sequences = len(filtered_sequences)

        self.normalized_sequences = max_value * np.ones(
            (n_sequences, sequence_len),
            dtype=np.int64,
        )

        for i, seq in enumerate(filtered_sequences):
            actual_len = min(len(seq), sequence_len)
            self.normalized_sequences[i, :actual_len] = seq[:actual_len]

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.normalized_sequences)

    def __getitem__(self, ix):
        """Get a single sample of the dataset."""
        return self.normalized_sequences[ix]


class Network(nn.Module):
    """Network predicting next number in the sequence.

    Parameters
    ----------
    max_value : int
        Maximum integer value allowed inside of the sequence. We will
        generate an embedding for each of the numbers in `[0, max_value]`.

    embedding_dim : int
        Dimensionality of the integer embeddings.

    n_layers : int
        Number of layers inside of the LSTM.

    hidden_dim : int
        Dimensionality of the hidden state (LSTM).

    dense_dim : int
        Dimensionality of the dense layer.

    Attributes
    ----------
    embedding : torch.nn.Embedding
        Embeddings of all the integers.

    lstm : torch.nn.LSTM
        LSTM subnetwork. Inputs integer embeddings and outputs
        new hidden states.

    linear : torch.nn.Linear
        Inputs hidden states and tranforms them.

    classifier : torch.nn.Linear
        Inputs outputs of the `linear` and outputs the logits
        over all possible integers.
    """

    def __init__(
            self,
            max_value=2000,
            embedding_dim=100,
            n_layers=2,
            hidden_dim=64,
            dense_dim=256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_value + 1,
            embedding_dim=embedding_dim,
            padding_idx=max_value,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(
            hidden_dim,
            dense_dim,
        )

        self.classifier = nn.Linear(
            dense_dim,
            max_value,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, sequence_len)` and has
            dtype `torch.long`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all possible integers of shape
            `(batch_size, sequence_len, max_value)`.
        """
        emb = self.embedding(x)  # (batch_size, sequence_len, embedding_dim)
        h, *_ = self.lstm(emb)  # (batch_size, sequence_len, hidden_dim)
        dense = torch.relu(
            self.linear(h)
        )  # (batch_size, sequence_len, dense_dim)
        logits = self.classifier(
            dense
        )  # (batch_size, sequence_len, max_value)

        return logits


def train_classifier(X, y, random_state=2):
    """Cross-validate classification problem using logistic regression.

    Parameters
    ----------
    X : np.ndarray
        2D array holding the features of shape `(n_samples, n_features)`.

    y : np.ndarray
        1D array holding the classification targets of shape `(n_samples,)`.

    random_state : int
        Guaranteeing reproducibility.

    Returns
    -------
    metrics : dict
        Holds train and validation accuracy averaged over all the folds.
    """
    cv = StratifiedKFold(
        n_splits=5,
        random_state=random_state,
        shuffle=True,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
    )

    res = cross_validate(
        clf,
        X,
        y,
        return_train_score=True,
        cv=cv,
    )

    metrics = {
        "train_acc": res["train_score"].mean(),
        "test_acc": res["test_score"].mean(),
    }

    return metrics


def create_classification_targets(indices):
    """Create multiple classification targets.

    They represent common properties of integers.

    Parameters
    ----------
    indices : np.ndarray
        1D array holding the integers for which we want to compute
        the targets.

    Returns
    -------
    targets : dict
        Keys are property names and the values are arrays of the same shape
        as `indices` representing whether a given integer does / does not
        have a given property.
    """

    targets = {
        "divisibility_2": (indices % 2 == 0).astype(float),
        "divisibility_3": (indices % 3 == 0).astype(float),
        "divisibility_4": (indices % 4 == 0).astype(float),
        "divisibility_5": (indices % 5 == 0).astype(float),
        "divisibility_10": (indices % 10 == 0).astype(float),
        "prime": np.vectorize(isprime)(indices).astype(float),
    }

    return targets
