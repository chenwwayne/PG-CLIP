import os
import unittest
from unittest import mock

import torch

import forward_utils


class ApsfWeightTest(unittest.TestCase):
    def setUp(self):
        self.prompt_mode = mock.patch.dict(
            os.environ, {"PROMPT_MODE": "apsf"}
        )
        self.prompt_mode.start()

    def tearDown(self):
        self.prompt_mode.stop()

    def test_tensor_weight_keeps_gradient(self):
        weight = torch.tensor(0.25, requires_grad=True)
        mvfa = torch.tensor([1.0, 0.0])
        pa = torch.tensor([0.0, 1.0])
        with mock.patch.object(
            forward_utils, "_encode_prompt_anchor", side_effect=[mvfa, pa]
        ):
            anchor = forward_utils._encode_state_anchor(
                object(),
                "AMOLED-Active",
                "object",
                1,
                torch.device("cpu"),
                apsf_weight=weight,
            )
        anchor.sum().backward()
        self.assertIsNotNone(weight.grad)
        self.assertNotEqual(float(weight.grad), 0.0)

    def test_zero_weight_reproduces_mvfa_anchor(self):
        mvfa = torch.tensor([1.0, 0.0])
        pa = torch.tensor([0.0, 1.0])
        with mock.patch.object(
            forward_utils, "_encode_prompt_anchor", side_effect=[mvfa, pa]
        ):
            anchor = forward_utils._encode_state_anchor(
                object(),
                "AMOLED-Active",
                "object",
                1,
                torch.device("cpu"),
                apsf_weight=torch.tensor(0.0),
            )
        torch.testing.assert_close(anchor, mvfa)

    def test_weight_must_be_bounded(self):
        mvfa = torch.tensor([1.0, 0.0])
        with mock.patch.object(
            forward_utils, "_encode_prompt_anchor", return_value=mvfa
        ):
            with self.assertRaisesRegex(ValueError, "must be in"):
                forward_utils._encode_state_anchor(
                    object(),
                    "AMOLED-Active",
                    "object",
                    1,
                    torch.device("cpu"),
                    apsf_weight=torch.tensor(1.1),
                )


if __name__ == "__main__":
    unittest.main()
