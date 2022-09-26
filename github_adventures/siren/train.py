import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import DataLoader

from core import GradientUtils, ImageSiren, PixelDataset

# Image loading
img_ = plt.imread("dog.png")
downsampling_factor = 4
img = 2 * (img_ - 0.5)
img = img[::downsampling_factor, ::downsampling_factor]
size = img.shape[0]

dataset = PixelDataset(img)

# Parameters
n_epochs = 100
batch_size = int(size ** 2)
logging_freq = 20

model_name = "siren"  # "siren", "mlp_relu"
hidden_features = 256
hidden_layers = 3

target = "intensity"  # "intensity", "grad", "laplace"

# Model creation
if model_name == "siren":
    model = ImageSiren(
        hidden_features,
        hidden_layers=hidden_layers,
        hidden_omega=30,
    )
elif model_name == "mlp_relu":
    layers = [Linear(2, hidden_features), ReLU()]

    for _ in range(hidden_layers):
        layers.append(Linear(hidden_features, hidden_features))
        layers.append(ReLU())

    layers.append(Linear(hidden_features, 1))

    model = Sequential(*layers)

    for module in model.modules():
        if not isinstance(module, Linear):
            continue
        torch.nn.init.xavier_normal_(module.weight)
else:
    raise ValueError("Unsupported model")

dataloader = DataLoader(dataset, batch_size=batch_size)
optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

# Training loop
for e in range(n_epochs):
    losses = []
    for d_batch in tqdm.tqdm(dataloader):
        x_batch = d_batch["coords"].to(torch.float32)
        x_batch.requires_grad = True

        y_true_batch = d_batch["intensity"].to(torch.float32)
        y_true_batch = y_true_batch[:, None]

        y_pred_batch = model(x_batch)

        if target == "intensity":
            loss = ((y_true_batch - y_pred_batch) ** 2).mean()

        elif target == "grad":
            y_pred_g_batch = GradientUtils.gradient(y_pred_batch, x_batch)
            y_true_g_batch = d_batch["grad"].to(torch.float32)
            loss = ((y_true_g_batch - y_pred_g_batch) ** 2).mean()

        elif target == "laplace":
            y_pred_l_batch = GradientUtils.laplace(y_pred_batch, x_batch)
            y_true_l_batch = d_batch["laplace"].to(torch.float32)[:, None]
            loss = ((y_true_l_batch - y_pred_l_batch) ** 2).mean()

        else:
            raise ValueError("Unrecognized target")

        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

    print(e, np.mean(losses))

    if e % logging_freq == 0:
        pred_img = np.zeros_like(img)
        pred_img_grad_norm = np.zeros_like(img)
        pred_img_laplace = np.zeros_like(img)

        orig_img = np.zeros_like(img)
        for d_batch in tqdm.tqdm(dataloader):
            coords = d_batch["coords"].to(torch.float32)
            coords.requires_grad = True
            coords_abs = d_batch["coords_abs"].numpy()

            pred = model(coords)
            pred_n = pred.detach().numpy().squeeze()
            pred_g = (
                GradientUtils.gradient(pred, coords)
                .norm(dim=-1)
                .detach()
                .numpy()
                .squeeze()
            )
            pred_l = GradientUtils.laplace(pred, coords).detach().numpy().squeeze()

            pred_img[coords_abs[:, 0], coords_abs[:, 1]] = pred_n
            pred_img_grad_norm[coords_abs[:, 0], coords_abs[:, 1]] = pred_g
            pred_img_laplace[coords_abs[:, 0], coords_abs[:, 1]] = pred_l

        fig, axs = plt.subplots(3, 2, constrained_layout=True)
        axs[0, 0].imshow(dataset.img, cmap="gray")
        axs[0, 1].imshow(pred_img, cmap="gray")

        axs[1, 0].imshow(dataset.grad_norm, cmap="gray")
        axs[1, 1].imshow(pred_img_grad_norm, cmap="gray")

        axs[2, 0].imshow(dataset.laplace, cmap="gray")
        axs[2, 1].imshow(pred_img_laplace, cmap="gray")

        for row in axs:
            for ax in row:
                ax.set_axis_off()

        fig.suptitle(f"Iteration: {e}")
        axs[0, 0].set_title("Ground truth")
        axs[0, 1].set_title("Prediction")

        plt.savefig(f"visualization/{e}.png")
