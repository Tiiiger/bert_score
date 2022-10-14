import unittest

from transformers import __version__ as ht_version

import bert_score
from tests.custom_assertions import CustomAssertions

cands = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was found dead in the staircase of a local shopping center.",
    'The victim\'s brother said he cannot imagine anyone who would want to harm him,"Finally, it went uphill again at him."',
]
refs = [
    "28-Year-Old Chef Found Dead at San Francisco Mall",
    "A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.",
    "But the victim's brother says he can't think of anyone who would want to hurt him, saying, \"Things were finally going well for him.\"",
]


class TestScore(unittest.TestCase, CustomAssertions):
    def test_scorer(self):
        scorer = bert_score.BERTScorer(lang="en", batch_size=3)

        (P, R, F), hash_code = scorer.score(cands, refs, return_hash=True)
        self.assertAreTensors(P, R, F)
        self.assertTensorsAlmostEqual(
            P, [0.9843302369117737, 0.9832239747047424, 0.9120386242866516]
        )
        self.assertTensorsAlmostEqual(
            R, [0.9823839068412781, 0.9732863903045654, 0.920428991317749]
        )
        self.assertTensorsAlmostEqual(
            F, [0.9833561182022095, 0.9782299995422363, 0.916214644908905]
        )
        self.assertEqual(
            hash_code,
            f"roberta-large_L17_no-idf_version={bert_score.__version__}(hug_trans={ht_version})",
        )

    def test_idf_scorer(self):
        scorer = bert_score.BERTScorer(
            lang="en", idf=True, idf_sents=refs, batch_size=3
        )

        (P, R, F), hash_code = scorer.score(cands, refs, return_hash=True)
        self.assertAreTensors(P, R, F)
        self.assertTensorsAlmostEqual(
            P, [0.9837872385978699, 0.9754738807678223, 0.8947395086288452]
        )
        self.assertTensorsAlmostEqual(
            R, [0.9827190637588501, 0.9697767496109009, 0.9172918796539307]
        )
        self.assertTensorsAlmostEqual(
            F, [0.9832529425621033, 0.972616970539093, 0.9058753848075867]
        )
        self.assertEqual(
            hash_code,
            f"roberta-large_L17_idf_version={bert_score.__version__}(hug_trans={ht_version})",
        )

    def test_scorer_rescale(self):
        scorer = bert_score.BERTScorer(
            lang="en", rescale_with_baseline=True, batch_size=3
        )

        (P, R, F), hash_code = scorer.score(cands, refs, return_hash=True)
        self.assertAreTensors(P, R, F)
        self.assertTensorsAlmostEqual(
            P, [0.907000780105591, 0.900435566902161, 0.477955609560013]
        )
        self.assertTensorsAlmostEqual(
            R, [0.895456790924072, 0.841467440128326, 0.527785062789917]
        )
        self.assertTensorsAlmostEqual(
            F, [0.901383399963379, 0.871010780334473, 0.503565192222595]
        )
        self.assertEqual(
            hash_code,
            f"roberta-large_L17_no-idf_version={bert_score.__version__}(hug_trans={ht_version})-rescaled",
        )

    def test_idf_scorer_rescale(self):
        scorer = bert_score.BERTScorer(
            lang="en",
            rescale_with_baseline=True,
            idf=True,
            idf_sents=refs,
            batch_size=3,
        )

        (P, R, F), hash_code = scorer.score(cands, refs, return_hash=True)
        self.assertAreTensors(P, R, F)
        self.assertTensorsAlmostEqual(
            P, [0.903778135776520, 0.854439020156860, 0.375287383794785]
        )
        self.assertTensorsAlmostEqual(
            R, [0.897446095943451, 0.820639789104462, 0.509167850017548]
        )
        self.assertTensorsAlmostEqual(
            F, [0.900772094726562, 0.837753534317017, 0.442304641008377]
        )
        self.assertEqual(
            hash_code,
            f"roberta-large_L17_idf_version={bert_score.__version__}(hug_trans={ht_version})-rescaled",
        )

    def test_multi_refs(self):
        scorer = bert_score.BERTScorer(
            lang="en", batch_size=3, rescale_with_baseline=True
        )

        cands = ["I like lemons."]
        refs = [["I am proud of you.", "I love lemons.", "Go go go."]]
        P_mul, R_mul, F_mul = scorer.score(
            cands,
            refs,
        )
        P_best, R_best, F_best = scorer.score(
            cands,
            [refs[0][1]],
        )
        self.assertTensorsAlmostEqual(P_mul, P_best)
        self.assertTensorsAlmostEqual(R_mul, R_best)
        self.assertTensorsAlmostEqual(F_mul, F_best)

    def test_multi_refs_working(self):
        scorer = bert_score.BERTScorer(
            lang="en", batch_size=3, rescale_with_baseline=True
        )

        cands = ["I like lemons.", "Hi", "Hey", "Hello", "Go", ""]
        refs = [
            ["I am proud of you.", "I love lemons.", "Go go go."],
            ["I am proud of you.", "Go go go."],
            ["Hi", ""],
            ["I am proud of you.", "I love lemons.", "Go go go.", "hello"],
            ["I am proud of you.", "Go go go.", "Go", "Go to school"],
            ["test"],
        ]
        P_mul, R_mul, F_mul = scorer.score(
            cands,
            refs,
        )
        self.assertAreTensors(P_mul, R_mul, F_mul)


if __name__ == "__main__":
    unittest.main()
