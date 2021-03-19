import pathlib
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from core import ImageSiren

torch.manual_seed(2)

init_functions = {
        "ones": torch.nn.init.ones_,
        "eye": torch.nn.init.eye_,
        "default": partial(torch.nn.init.kaiming_uniform_, a=5 ** (1 / 2)),
        "paper": None,
}

for fname, func in init_functions.items():
    path = pathlib.Path.cwd() / "tensorboard_logs" / fname
    writer = SummaryWriter(path)

    def fh(inst, inp, out, number=0):
        layer_name = f"{number}_{inst.__class__.__name__}"
        writer.add_histogram(layer_name, out)

    model = ImageSiren(
            hidden_layers=10,
            hidden_features=200,
            first_omega=30,
            hidden_omega=30,
            custom_init_function_=func,
    )

    for i, layer in enumerate(model.net.modules()):
        if not i:
            continue
        layer.register_forward_hook(partial(fh, number=(i + 1) // 2))

    inp = 2 * (torch.rand(10000, 2) - 0.5)
    writer.add_histogram("0", inp)
    res = model(inp)
