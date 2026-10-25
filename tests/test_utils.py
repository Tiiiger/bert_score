import unittest

import torch

from bert_score.utils import greedy_cos_idf
from tests.custom_assertions import CustomAssertions


class TestGreedyCosIdf(unittest.TestCase, CustomAssertions):
    def test_empty_inputs_with_positive_special_token_idf(self):
        for all_layers in (False, True):
            with self.subTest(all_layers=all_layers):
                # Empty hypothesis, empty reference, and a nonempty identical pair.
                ref_masks = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1]])
                hyp_masks = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1]])
                shape = (3, 3, 2, 4) if all_layers else (3, 3, 4)
                scores = greedy_cos_idf(
                    torch.ones(shape),
                    ref_masks,
                    ref_masks.float(),
                    torch.ones(shape),
                    hyp_masks,
                    hyp_masks.float(),
                    all_layers=all_layers,
                )
                expected = [[0, 0, 1], [0, 0, 1]] if all_layers else [0, 0, 1]
                for score in scores:
                    self.assertTensorsAlmostEqual(score, expected)
