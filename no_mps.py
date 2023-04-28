import torch
from torch import mps
from time import perf_counter

# Create a Tensor directly on the mps device
mps_device = torch.device("mps")
print(mps.current_allocated_memory())
x = torch.ones(1000000000, device=mps_device)
w = torch.ones(1000000000, device=mps_device)
print(mps.current_allocated_memory() / 1e9)

# Any operation happens on the GPU
start = perf_counter()
y = x @ w
end = perf_counter()
elapsed_time = end - start
print(f"Y = {y}")
print(f"The computation took {elapsed_time}")