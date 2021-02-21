from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn import Embedding, Linear, LSTM, Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm


class CharacterDataset(Dataset):
    """Custom dataset.

    Parameters
    ----------
    text : str
        Input text that will be used to create the entire database.

    window_size : int
        Number of characters to use as input features.

    vocab_size : int
        Number of characters in the vocabulary. Note that the last character
        is always reserved for a special "~" out-of-vocabulary character.

    Attributes
    ----------
    ch2ix : defaultdict
        Mapping from the character to the position of that character in the
        vocabulary. Note that all characters that are not in the vocabulary
        will get mapped into the index `vocab_size - 1`.

    ix2ch : dict
        Mapping from the character position in the vocabulary to the actual
        character.

    vocabulary : list
        List of all characters. `len(vocabulary) == vocab_size`.
    """
    def __init__(self, text, window_size=1, vocab_size=50):
        self.text = text.replace("\n", " ")
        self.window_size = window_size
        self.ch2ix = defaultdict(lambda: vocab_size - 1)

        most_common_ch2ix = {
            x[0]: i
            for i, x in enumerate(Counter(self.text).most_common()[: (vocab_size - 1)])
        }
        self.ch2ix.update(most_common_ch2ix)
        self.ch2ix["~"] = vocab_size - 1

        self.ix2ch = {v: k for k, v in self.ch2ix.items()}
        self.vocabulary = [self.ix2ch[i] for i in range(vocab_size)]

    def __len__(self):
        return len(self.text) - self.window_size

    def __getitem__(self, ix):
        X = torch.LongTensor(
            [self.ch2ix[c] for c in self.text[ix : ix + self.window_size]]
        )
        y = self.ch2ix[self.text[ix + self.window_size]]

        return X, y


class Network(Module):
    """Custom network predicting the next character of a string.

    Parameters
    ----------
    vocab_size : int
        The number of charactesr in the vocabulary.

    embedding_dim : int
        Dimension of the character embedding vectors.

    dense_dim : int
        Number of neurons in the linear layer that follows the LSTM.

    hidden_dim : int
        Size of the LSTM hidden state.

    max_norm : int
        If any of the embedding vectors has a higher L2 norm than `max_norm`
        it is rescaled.

    n_layers : int
        Number of the layers of the LSTM.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=2,
        dense_dim=32,
        hidden_dim=8,
        max_norm=2,
        n_layers=1,
    ):
        super().__init__()

        self.embedding = Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=vocab_size - 1,
                norm_type=2,
                max_norm=max_norm,
        )
        self.lstm = LSTM(
                embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers
        )
        self.linear_1 = Linear(hidden_dim, dense_dim)
        self.linear_2 = Linear(dense_dim, vocab_size)


    def forward(self, x, h=None, c=None):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, window_size)` of dtype
            `torch.int64`.

        h, c : torch.Tensor or None
            Hidden states of the LSTM.

        Returns
        -------
        logits : torch.Tensor
            Tensor of shape `(n_samples, vocab_size)`.

        h, c : torch.Tensor or None
            Hidden states of the LSTM.
        """
        emb = self.embedding(x)  # (n_samples, window_size, embedding_dim)
        if h is not None and c is not None:
            _, (h, c) = self.lstm(emb, (h, c))
        else:
            _, (h, c) = self.lstm(emb)  # (n_layers, n_samples, hidden_dim)

        h_mean = h.mean(dim=0)  # (n_samples, hidden_dim)
        x = self.linear_1(h_mean)  # (n_samples, dense_dim)
        logits = self.linear_2(x)  # (n_samples, vocab_size)

        return logits, h, c

def compute_loss(cal, net, dataloader):
    """Computer average loss over a dataset."""
    net.eval()
    all_losses = []
    for X_batch, y_batch in dataloader:
        probs, _, _ = net(X_batch)

        all_losses.append(cal(probs, y_batch).item())

    return np.mean(all_losses)

def generate_text(n_chars, net, dataset, initial_text="Hello", random_state=None):
    """Generate text with the character-level model.

    Parameters
    ----------
    n_chars : int
        Number of characters to generate.

    net : Module
        Character-level model.

    dataset : CharacterDataset
        Instance of the `CharacterDataset`.

    initial_text : str
        The starting text to be used as the initial condition for the model.

    random_state : None or int
        If not None, then the result is reproducible.

    Returns
    -------
    res : str
        Generated text.
    """
    if not initial_text:
        raise ValueError("You need to specify the initial text")

    res = initial_text
    net.eval()
    h, c = None, None

    if random_state is not None:
        np.random.seed(random_state)

    for _ in range(n_chars):
        previous_chars = initial_text if res == initial_text else res[-1]
        features = torch.LongTensor([[dataset.ch2ix[c] for c in previous_chars]])
        logits, h, c = net(features, h, c)
        probs = F.softmax(logits[0], dim=0).detach().numpy()
        new_ch = np.random.choice(dataset.vocabulary, p=probs)
        res += new_ch

    return res

if __name__ == "__main__":
    with open("text.txt", "r") as f:
        text = "\n".join(f.readlines())

    # Hyperparameters model
    vocab_size = 70
    window_size = 10
    embedding_dim = 2
    hidden_dim = 16
    dense_dim = 32
    n_layers = 1
    max_norm = 2

    # Training config
    n_epochs = 25
    train_val_split = 0.8
    batch_size = 128
    random_state = 13

    torch.manual_seed(random_state)

    loss_f = torch.nn.CrossEntropyLoss()
    dataset = CharacterDataset(text, window_size=window_size, vocab_size=vocab_size)

    n_samples = len(dataset)
    split_ix = int(n_samples * train_val_split)

    train_indices, val_indices = np.arange(split_ix), np.arange(split_ix, n_samples)

    train_dataloader = DataLoader(
            dataset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size
    )
    val_dataloader = DataLoader(
            dataset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size
    )

    net = Network(
            vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dense_dim=dense_dim,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
    )
    optimizer = torch.optim.Adam(
            net.parameters(),
            lr=1e-2,
    )

    emb_history = []

    for e in range(n_epochs + 1):
        net.train()
        for X_batch, y_batch in tqdm(train_dataloader):
            if e == 0:
                break

            optimizer.zero_grad()
            probs, _, _ = net(X_batch)
            loss = loss_f(probs, y_batch)
            loss.backward()

            optimizer.step()

        train_loss = compute_loss(loss_f, net, train_dataloader)
        val_loss = compute_loss(loss_f, net, val_dataloader)
        print(f"Epoch: {e}, {train_loss=:.3f}, {val_loss=:.3f}")

        # Generate one sentence
        initial_text = "I hope it works "
        generated_text = generate_text(
            100, net, dataset, initial_text=initial_text, random_state=random_state
        )
        print(generated_text)

        # Prepare DataFrame
        weights = net.embedding.weight.detach().clone().numpy()

        df = pd.DataFrame(weights, columns=[f"dim_{i}" for i in range(embedding_dim)])
        df["epoch"] = e
        df["character"] = dataset.vocabulary

        emb_history.append(df)

final_df = pd.concat(emb_history)
final_df.to_csv("res.csv", index=False)
