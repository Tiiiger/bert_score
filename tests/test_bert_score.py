import unittest
import torch
import bert_score

EPS = 1e-6

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
        (P, R, F), hash_code = bert_score.score(cands, refs, model_type='roberta-large', num_layers=17,
                                              idf=False, batch_size=3, return_hash=True)
        print(P.tolist(), R.tolist(), F.tolist())

        self.assertTrue(torch.is_tensor(P))
        self.assertTrue(torch.is_tensor(R))
        self.assertTrue(torch.is_tensor(F))
        self.assertEqual(hash_code, f'roberta-large_L17_no-idf_version={bert_score.__version__}')
        self.assertTrue((P - torch.tensor([0.9843302369117737, 0.9832239747047424, 0.9120386242866516])).abs_().max() < EPS)
        self.assertTrue((R - torch.tensor([0.9823839068412781, 0.9732863903045654, 0.920428991317749])).abs_().max() < EPS)
        self.assertTrue((F - torch.tensor([0.9833561182022095, 0.9782299995422363, 0.916214644908905])).abs_().max() < EPS)

    def test_idf_score(self):
        (P, R, F), hash_code = bert_score.score(cands, refs, model_type='roberta-large', num_layers=17,
                                              idf=True, batch_size=3, return_hash=True)
        print(P.tolist(), R.tolist(), F.tolist())

        self.assertTrue(torch.is_tensor(P))
        self.assertTrue(torch.is_tensor(R))
        self.assertTrue(torch.is_tensor(F))
        self.assertEqual(hash_code, f'roberta-large_L17_idf_version={bert_score.__version__}')
        self.assertTrue((P - torch.tensor([0.9837872385978699, 0.9754738807678223, 0.8947395086288452])).abs_().max() < EPS)
        self.assertTrue((R - torch.tensor([0.9827190637588501, 0.9697767496109009, 0.9172918796539307])).abs_().max() < EPS)
        self.assertTrue((F - torch.tensor([0.9832529425621033, 0.972616970539093, 0.9058753848075867])).abs_().max() < EPS)

if __name__ == '__main__':
    unittest.main()
