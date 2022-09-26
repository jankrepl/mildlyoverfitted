import einops
import torch.nn as nn


class MlpBlock(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    dim : int
        Input and output dimension of the entire block. Inside of the mixer
        it will either be equal to `n_patches` or `hidden_dim`.

    mlp_dim : int
        Dimension of the hidden layer.

    Attributes
    ----------
    linear_1, linear_2 : nn.Linear
        Linear layers.

    activation : nn.GELU
        Activation.
    """

    def __init__(self, dim, mlp_dim=None):
        super().__init__()

        mlp_dim = dim if mlp_dim is None else mlp_dim
        self.linear_1 = nn.Linear(dim, mlp_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, n_channels, n_patches)` or
            `(n_samples, n_patches, n_channels)`.

        Returns
        -------
        torch.Tensor
            Output tensor that has exactly the same shape as the input `x`.
        """
        x = self.linear_1(x)  # (n_samples, *, mlp_dim)
        x = self.activation(x)  # (n_samples, *, mlp_dim)
        x = self.linear_2(x)  # (n_samples, *, dim)
        return x


class MixerBlock(nn.Module):
    """Mixer block that contains two `MlpBlock`s and two `LayerNorm`s.

    Parameters
    ----------
    n_patches : int
        Number of patches the image is split up into.

    hidden_dim : int
        Dimensionality of patch embeddings.

    tokens_mlp_dim : int
        Hidden dimension for the `MlpBlock` when doing token mixing.

    channels_mlp_dim : int
        Hidden dimension for the `MlpBlock` when doing channel mixing.

    Attributes
    ----------
    norm_1, norm_2 : nn.LayerNorm
        Layer normalization.

    token_mlp_block : MlpBlock
        Token mixing MLP.

    channel_mlp_block : MlpBlock
        Channel mixing MLP.
    """

    def __init__(
            self, *, n_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim
    ):
        super().__init__()

        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)

        self.token_mlp_block = MlpBlock(n_patches, tokens_mlp_dim)
        self.channel_mlp_block = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_patches, hidden_dim)`.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as `x`, i.e.
            `(n_samples, n_patches, hidden_dim)`.
        """
        y = self.norm_1(x)  # (n_samples, n_patches, hidden_dim)
        y = y.permute(0, 2, 1)  # (n_samples, hidden_dim, n_patches)
        y = self.token_mlp_block(y)  # (n_samples, hidden_dim, n_patches)
        y = y.permute(0, 2, 1)  # (n_samples, n_patches, hidden_dim)
        x = x + y  # (n_samples, n_patches, hidden_dim)
        y = self.norm_2(x)  # (n_samples, n_patches, hidden_dim)
        res = x + self.channel_mlp_block(
            y
        )  # (n_samples, n_patches, hidden_dim)
        return res


class MlpMixer(nn.Module):
    """Entire network.

    Parameters
    ----------
    image_size : int
        Height and width (assuming it is a square) of the input image.

    patch_size : int
        Height and width (assuming it is a square) of the patches. Note
        that we assume that `image_size % patch_size == 0`.

    tokens_mlp_dim : int
        Hidden dimension for the `MlpBlock` when doing the token mixing.

    channels_mlp_dim : int
        Hidden dimension for the `MlpBlock` when diong the channel mixing.

    n_classes : int
        Number of classes for classification.

    hidden_dim : int
        Dimensionality of patch embeddings.

    n_blocks : int
        The number of `MixerBlock`s in the architecture.

    Attributes
    ----------
    patch_embedder : nn.Conv2D
        Splits the image up into multiple patches and then embeds each of them
        (using shared weights).

    blocks : nn.ModuleList
        List of `MixerBlock` instances.

    pre_head_norm : nn.LayerNorm
        Layer normalization applied just before the classification head.

    head_classifier : nn.Linear
        The classification head.
    """

    def __init__(
            self,
            *,
            image_size,
            patch_size,
            tokens_mlp_dim,
            channels_mlp_dim,
            n_classes,
            hidden_dim,
            n_blocks,
    ):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2  # assumes divisibility

        self.patch_embedder = nn.Conv2d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    n_patches=n_patches,
                    hidden_dim=hidden_dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                )
                for _ in range(n_blocks)
            ]
        )

        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.head_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of square images of shape
            `(n_samples, n_channels, image_size, image_size)`.

        Returns
        -------
        torch.Tensor
            Class logits of shape `(n_samples, n_classes)`.
        """
        x = self.patch_embedder(
            x
        )  # (n_samples, hidden_dim, n_patches ** (1/2), n_patches ** (1/2))
        x = einops.rearrange(
            x, "n c h w -> n (h w) c"
        )  # (n_samples, n_patches, hidden_dim)
        for mixer_block in self.blocks:
            x = mixer_block(x)  # (n_samples, n_patches, hidden_dim)

        x = self.pre_head_norm(x)  # (n_samples, n_patches, hidden_dim)
        x = x.mean(dim=1)  # (n_samples, hidden_dim)
        y = self.head_classifier(x)  # (n_samples, n_classes)

        return y
