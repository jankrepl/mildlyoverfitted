import torch

bn = torch.nn.BatchNorm1d(5)
bn.state_dict()

for name, p in bn.named_buffers():
    print(name, p, p.requires_grad)

for name, p in bn.named_parameters():
    print(name, p, p.requires_grad)
