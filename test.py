import torch
from torch.profiler import profile, ProfilerActivity


def demo_basic_profiling():
    device = torch.device("openreg")
    input_data = torch.randn(32, 1, 28, 28).to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        x = input_data**2

    print(prof.key_averages().table())


demo_basic_profiling()
