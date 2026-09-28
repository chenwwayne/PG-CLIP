import unittest

import numpy as np
import torch

from forward_utils import (
    abnormal_anchor_image_score,
    combine_industrial_image_scores,
    minmax_normalize,
)


class ScoringTest(unittest.TestCase):
    def test_abnormal_anchor_image_score(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        anchors = torch.eye(2)
        torch.testing.assert_close(
            abnormal_anchor_image_score(features, anchors),
            torch.tensor([0.5, 1.0]),
        )

    def test_industrial_score_fuses_pixel_max_and_image_score(self):
        pixel = np.array([[[0.1, 0.8]], [[0.2, 0.4]]])
        image = np.array([0.6, 0.2])
        np.testing.assert_allclose(
            combine_industrial_image_scores(pixel, image), [0.7, 0.3]
        )

    def test_constant_minmax_scores_are_finite(self):
        result = minmax_normalize(np.ones((2, 3)))
        np.testing.assert_array_equal(result, np.zeros((2, 3)))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
