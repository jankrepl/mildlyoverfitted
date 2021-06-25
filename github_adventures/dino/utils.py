import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class DataAugmentation:
    """Create crops of an input image together with additional augmentation.

    It generates 2 global crops and `n_local_crops` local crops.

    Parameters
    ----------
    global_crops_scale : tuple
        Range of sizes for the global crops.

    local_crops_scale : tuple
        Range of sizes for the local crops.

    n_local_crops : int
        Number of local crops to create.

    size : int
        The size of the final image.

    Attributes
    ----------
    global_1, global_2 : transforms.Compose
        Two global transforms.

    local : transforms.Compose
        Local transform. Note that the augmentation is stochastic so one
        instance is enough and will lead to different crops.
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1),
        local_crops_scale=(0.05, 0.4),
        n_local_crops=8,
        size=224,
    ):
        self.n_local_crops = n_local_crops
        RandomGaussianBlur = lambda p: transforms.RandomApply(  # noqa
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))],
            p=p,
        )

        flip_and_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        ),
                    ]
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(1.0),  # always apply
                normalize,
            ],
        )

        self.global_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                transforms.RandomSolarize(170, p=0.2),
                normalize,
            ],
        )

        self.local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.5),
                normalize,
            ],
        )

    def __call__(self, img):
        """Apply transformation.

        Parameters
        ----------
        img : PIL.Image
            Input image.

        Returns
        -------
        all_crops : list
            List of `torch.Tensor` representing different views of
            the input `img`.
        """
        all_crops = []
        all_crops.append(self.global_1(img))
        all_crops.append(self.global_2(img))

        all_crops.extend([self.local(img) for _ in range(self.n_local_crops)])

        return all_crops


class Head(nn.Module):
    """Network hooked up to the CLS token embedding.

    Just a MLP with the last layer being normalized in a particular way.

    Parameters
    ----------
    in_dim : int
        The dimensionality of the token embedding.

    out_dim : int
        The dimensionality of the final layer (we compute the softmax over).

    hidden_dim : int
        Dimensionality of the hidden layers.

    bottleneck_dim : int
        Dimensionality of the second last layer.

    n_layers : int
        The number of layers.

    norm_last_layer : bool
        If True, then we freeze the norm of the weight of the last linear layer
        to 1.

    Attributes
    ----------
    mlp : nn.Sequential
        Vanilla multi-layer perceptron.

    last_layer : nn.Linear
        Reparametrized linear layer with weight normalization. That means
        that that it will have `weight_g` and `weight_v` as learnable
        parameters instead of a single `weight`.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=512,
        bottleneck_dim=256,
        n_layers=3,
        norm_last_layer=False,
    ):
        super().__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """Initialize learnable parameters."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, in_dim)`.

        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, out_dim)`.
        """
        x = self.mlp(x)  # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)  # (n_samples, bottleneck_dim)
        x = self.last_layer(x)  # (n_samples, out_dim)

        return x


class MultiCropWrapper(nn.Module):
    """Convenience class for forward pass of multiple crops.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Instantiated Vision Transformer. Note that we will take the `head`
        attribute and replace it with `nn.Identity`.

    new_head : Head
        New head that is going to be put on top of the `backbone`.
    """
    def __init__(self, backbone, new_head):
        super().__init__()
        backbone.head = nn.Identity()  # desactivate original head
        self.backbone = backbone
        self.new_head = new_head

    def forward(self, x):
        """Run the forward pass.

        The different crops are concatenated along the batch dimension
        and then a single forward pass is fun. The resulting tensor
        is then chunked back to per crop tensors.

        Parameters
        ----------
        x : list
            List of `torch.Tensor` each of shape `(n_samples, 3, size, size)`.

        Returns
        -------
        tuple
            Tuple of `torch.Tensor` each of shape `(n_samples, out_dim)` where
            `output_dim` is determined by `Head`.
        """
        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)  # (n_samples * n_crops, 3, size, size)
        cls_embedding = self.backbone(concatenated)  # (n_samples * n_crops, in_dim)
        logits = self.new_head(cls_embedding)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)

        return chunks


class Loss(nn.Module):
    """The loss function.

    We subclass the `nn.Module` becuase we want to create a buffer for the
    logits center of the teacher.

    Parameters
    ----------
    out_dim : int
        The dimensionality of the final layer (we computed the softmax over).

    teacher_temp, student_temp : float
        Softmax temperature of the teacher resp. student.

    center_momentum : float
        Hyperparameter for the exponential moving average that determines
        the center logits. The higher the more the running average matters.
    """
    def __init__(
        self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """Evaluate loss.

        Parameters
        ----------
        student_output, teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing
            logits. The length is equal to number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the average loss.
        """
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
                total_loss += loss.mean()  # scalar
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients.

    Parameters
    ----------
    model : nn.Module
        Module.

    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
