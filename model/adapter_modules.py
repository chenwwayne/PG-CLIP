import random
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


@contextmanager
def preserve_global_rng_state():
    """Keep optional module construction from perturbing application RNG streams."""
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def initialize_linear_layers(module):
    """Apply the image branch's historical Xavier initialization policy."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class SimpleAdapter(nn.Module):
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleProj(nn.Module):
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


def _build_pykan(width, grid_size, spline_order, seed):
    """Build pykan without letting its constructor reset application RNG state."""
    try:
        from kan import KAN
    except ImportError as exc:
        raise ImportError(
            "KAN image modules require pykan==0.2.8; install project requirements"
        ) from exc

    with preserve_global_rng_state():
        model = KAN(
            width=list(width),
            grid=grid_size,
            k=spline_order,
            seed=seed,
            symbolic_enabled=False,
            save_act=False,
            auto_save=False,
            device="cpu",
        )
        model.speed()
    return model


class PyKANLayer(nn.Module):
    """Shape-preserving wrapper around the standard pykan KAN module."""

    def __init__(self, width, grid_size=3, spline_order=3, seed=0):
        super().__init__()
        self.width = width
        self.kan = _build_pykan(
            [width, width],
            grid_size=grid_size,
            spline_order=spline_order,
            seed=seed,
        )

    def forward(self, x):
        shape = x.shape
        if shape[-1] != self.width:
            raise ValueError(f"expected final dimension {self.width}, got {shape[-1]}")
        parameter = next(self.kan.parameters())
        flat = x.reshape(-1, self.width).to(
            device=parameter.device, dtype=parameter.dtype
        )
        output = self.kan(flat)
        return output.to(dtype=x.dtype).reshape(shape)


class KANAdapter(nn.Module):
    """Lightweight bottleneck adapter with a standard pykan numerical core."""

    def __init__(
        self,
        c_in,
        c_out=768,
        bottleneck=16,
        grid_size=3,
        spline_order=3,
        seed=0,
        relu=False,
    ):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("KAN bottleneck must be positive")
        self.c_in = c_in
        self.c_out = c_out
        with preserve_global_rng_state():
            layers = [
                nn.Linear(c_in, bottleneck, bias=False),
                PyKANLayer(bottleneck, grid_size, spline_order, seed),
                nn.Linear(bottleneck, c_out, bias=False),
            ]
            if relu:
                layers.append(nn.LeakyReLU())
            self.fc = nn.Sequential(*layers)
        # Advance the global stream exactly like the single Linear this replaces.
        nn.Linear(c_in, c_out, bias=False)

    def initialize_preserving_replacement_rng(self):
        """Initialize KAN while consuming one baseline Linear's Xavier draw."""
        baseline_weight = torch.empty(self.c_out, self.c_in)
        nn.init.xavier_uniform_(baseline_weight)
        with preserve_global_rng_state():
            initialize_linear_layers(self)

    def forward(self, x):
        return self.fc(x)


class ResidualKANAdapter(nn.Module):
    """Baseline-preserving simple path plus a learnable KAN correction."""

    def __init__(
        self,
        c_in,
        c_out=768,
        bottleneck=16,
        grid_size=3,
        spline_order=3,
        seed=0,
        relu=True,
    ):
        super().__init__()
        self.base = SimpleProj(c_in, c_out, relu=relu)
        with preserve_global_rng_state():
            self.correction = KANAdapter(
                c_in,
                c_out,
                bottleneck=bottleneck,
                grid_size=grid_size,
                spline_order=spline_order,
                seed=seed,
                relu=relu,
            )
        self.correction_scale = nn.Parameter(torch.zeros(()))

    def initialize_preserving_base_rng(self):
        """Initialize the base exactly as simple, isolating correction RNG use."""
        initialize_linear_layers(self.base)
        with preserve_global_rng_state():
            initialize_linear_layers(self.correction)

    def forward(self, x):
        return self.base(x) + torch.tanh(self.correction_scale) * self.correction(x)
