import unittest
import torch
import bert_score

from collections import defaultdict

eps = 1e-6

cands = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was found dead in the staircase of a local shopping center.",
    "The victim's brother said he cannot imagine anyone who would want to harm him,\"Finally, it went uphill again at him.\"",
]
refs = [
    "28-Year-Old Chef Found Dead at San Francisco Mall",
    "A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.",
    "But the victim's brother says he can't think of anyone who would want to hurt him, saying, \"Things were finally going well for him.\""
]

class TestScore(unittest.TestCase):
    def test_score(self):
        P, R, F, hash_code = bert_score.score(cands, refs, model_type='roberta-large', num_layers=17,
                                              idf=False, batch_size=3, return_hash=True)
        print(P.tolist(), R.tolist(), F.tolist())

        self.assertTrue(torch.is_tensor(P))
        self.assertTrue(torch.is_tensor(R))
        self.assertTrue(torch.is_tensor(F))
        self.assertEqual(hash_code, f'roberta-large_L17_no-idf_version={bert_score.__version__}')
        self.assertTrue((P - torch.tensor([0.9862896203994751, 0.9817618131637573, 0.9145744442939758])).abs_().max() < eps)
        self.assertTrue((R - torch.tensor([0.986611008644104, 0.9717907905578613, 0.9223880767822266])).abs_().max() < eps)
    def test_idf_score(self):
        P, R, F, hash_code = bert_score.score(cands, refs, model_type='roberta-large', num_layers=17,
                                              idf=True, batch_size=3, return_hash=True)
        print(P.tolist(), R.tolist(), F.tolist())

        self.assertTrue(torch.is_tensor(P))
        self.assertTrue(torch.is_tensor(R))
        self.assertTrue(torch.is_tensor(F))
        self.assertEqual(hash_code, f'roberta-large_L17_idf_version={bert_score.__version__}')
        self.assertTrue((P - torch.tensor([0.9841673374176025, 0.9752232432365417, 0.8989502787590027])).abs_().max() < eps)
        self.assertTrue((R - torch.tensor([0.9843330979347229, 0.9698787927627563, 0.9181708097457886])).abs_().max() < eps)
        self.assertTrue((F - torch.tensor([0.9842502474784851, 0.9725437164306641, 0.908458948135376])).abs_().max() < eps)

if __name__ == '__main__':
    unittest.main()
