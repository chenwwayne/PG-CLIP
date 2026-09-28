import random
import unittest

import numpy as np
import torch

from model.adapter_modules import (
    KANAdapter,
    PyKANLayer,
    ResidualKANAdapter,
    SimpleAdapter,
    initialize_linear_layers,
)


class KANAdapterTest(unittest.TestCase):
    def test_pykan_layer_preserves_shape_and_backpropagates(self):
        layer = PyKANLayer(4, grid_size=2, spline_order=2, seed=0)
        x = torch.randn(3, 5, 4, requires_grad=True)

        output = layer(x)
        output.square().mean().backward()

        self.assertEqual(output.shape, x.shape)
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(output).all())

    def test_bottleneck_adapter_supports_projection(self):
        adapter = KANAdapter(
            8, 6, bottleneck=4, grid_size=2, spline_order=2, seed=1
        )
        output = adapter(torch.randn(2, 3, 8))
        self.assertEqual(output.shape, (2, 3, 6))

    def test_residual_adapter_starts_as_simple_path_and_learns_scale(self):
        adapter = ResidualKANAdapter(
            8, 6, bottleneck=4, grid_size=2, spline_order=2, seed=3
        )
        x = torch.randn(2, 3, 8)

        output = adapter(x)
        expected = adapter.base(x)
        output.square().mean().backward()

        self.assertTrue(torch.equal(output, expected))
        self.assertIsNotNone(adapter.correction_scale.grad)
        self.assertTrue(torch.isfinite(adapter.correction_scale.grad))

    def test_residual_sequence_preserves_simple_base_initialization(self):
        torch.manual_seed(123)
        simple = [SimpleAdapter(8, 8) for _ in range(3)]
        for module in simple:
            initialize_linear_layers(module)
        simple_rng_state = torch.random.get_rng_state().clone()

        torch.manual_seed(123)
        residual = [
            ResidualKANAdapter(
                8, 8, bottleneck=4, grid_size=2, spline_order=2, seed=index
            )
            for index in range(3)
        ]
        for module in residual:
            module.initialize_preserving_base_rng()

        for simple_module, residual_module in zip(simple, residual):
            self.assertTrue(
                torch.equal(simple_module.fc[0].weight, residual_module.base.fc[0].weight)
            )
        self.assertTrue(torch.equal(simple_rng_state, torch.random.get_rng_state()))

    def test_direct_kan_consumes_same_rng_as_replaced_linear(self):
        torch.manual_seed(456)
        simple = [SimpleAdapter(8, 6) for _ in range(3)]
        for module in simple:
            initialize_linear_layers(module)
        simple_rng_state = torch.random.get_rng_state().clone()

        torch.manual_seed(456)
        kan = [
            KANAdapter(8, 6, bottleneck=4, grid_size=2, spline_order=2, seed=index)
            for index in range(3)
        ]
        for module in kan:
            module.initialize_preserving_replacement_rng()

        self.assertTrue(torch.equal(simple_rng_state, torch.random.get_rng_state()))

    def test_pykan_construction_preserves_global_rng_states(self):
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()

        PyKANLayer(3, grid_size=2, spline_order=2, seed=7)

        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        current_numpy_state = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy_state[0])
        self.assertTrue(np.array_equal(numpy_state[1], current_numpy_state[1]))
        self.assertEqual(numpy_state[2:], current_numpy_state[2:])
        self.assertEqual(python_state, random.getstate())


if __name__ == "__main__":
    unittest.main()
