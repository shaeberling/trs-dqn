"""Small, synchronized benchmark of the intended convolutional workload."""

import json
import time

import numpy as np


def torch_bench(device, threads=1):
    import torch
    from torch import nn

    torch.set_num_threads(threads)
    model = nn.Sequential(
        nn.Conv2d(4, 16, 5, 2, 2), nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(),
        nn.Linear(32 * 6 * 16, 256), nn.ReLU(), nn.Linear(256, 6),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.zeros((64, 4, 48, 128), device=device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        sync()

    for _ in range(5):
        step()
    start = time.perf_counter()
    for _ in range(40):
        step()
    train_ms = (time.perf_counter() - start) * 25
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(200):
            model(x[:1]).argmax().item()
        infer_ms = (time.perf_counter() - start) * 5
    return dict(backend=f"torch-{device}-{threads}", train_ms=train_ms, infer_ms=infer_ms)


def mlx_bench():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = [nn.Conv2d(4, 16, 5, 2, 2), nn.Conv2d(16, 32, 3, 2, 1),
                         nn.Conv2d(32, 32, 3, 2, 1)]
            self.fc = nn.Linear(32 * 6 * 16, 256)
            self.out = nn.Linear(256, 6)

        def __call__(self, x):
            for c in self.conv:
                x = nn.relu(c(x))
            return self.out(nn.relu(self.fc(x.reshape(x.shape[0], -1))))

    model = Net()
    opt = optim.Adam(1e-4)
    opt.init(model.trainable_parameters())
    state = [model.state, opt.state]
    vg = nn.value_and_grad(model, lambda m, x: mx.mean(m(x)**2))

    def step(x):
        loss, grads = vg(model, x)
        opt.update(model, grads)
        return loss

    step = mx.compile(step, inputs=state, outputs=state)
    infer = mx.compile(lambda x: mx.argmax(model(x), axis=1), inputs=model.state)
    x = mx.zeros((64, 48, 128, 4))
    for _ in range(5):
        mx.eval(step(x), state)
    start = time.perf_counter()
    for _ in range(40):
        mx.eval(step(x), state)
    train_ms = (time.perf_counter() - start) * 25
    infer(x[:1]).item()
    start = time.perf_counter()
    for _ in range(200):
        infer(x[:1]).item()
    return dict(backend="mlx-gpu-compiled", train_ms=train_ms,
                infer_ms=(time.perf_counter()-start)*5)


if __name__ == "__main__":
    for backend, threads in (("cpu", 1), ("cpu", 4), ("mps", 1)):
        print(json.dumps(torch_bench(backend, threads)), flush=True)
    print(json.dumps(mlx_bench()), flush=True)
