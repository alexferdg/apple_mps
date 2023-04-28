import torch
from torchdistx.fake import fake_mode

# Meta tensors are always "allocated" on the `meta` device.
a = torch.ones([10], device="meta")
a.device()