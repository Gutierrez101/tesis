import unittest

from model_comparison import resolve_temporal_consensus


def sample(predictions):
    return {
        name: (identity, 0.8 if identity != "Desconocido" else 0.4, 1.0)
        for name, identity in predictions.items()
    }


class TemporalReIDConsensusTests(unittest.TestCase):
    def test_requires_stable_architecture_predictions_and_three_model_votes(self):
        samples = [
            sample({
                "osnet_x1_0": "Ana", "repvit": "Ana",
                "efficientformerv2": "Ana", "fastvit": "Desconocido",
                "swiftformer": "Desconocido",
            })
            for _ in range(3)
        ]

        identity, confidence, votes = resolve_temporal_consensus(samples)

        self.assertEqual(identity, "Ana")
        self.assertAlmostEqual(confidence, 0.8)
        self.assertEqual(sum(vote == "Ana" for vote in votes.values()), 3)

    def test_rejects_unstable_model_and_architecture_tie(self):
        samples = [
            sample({
                "osnet_x1_0": "Ana", "repvit": "Ana",
                "efficientformerv2": "Ana", "fastvit": "Beatriz",
                "swiftformer": "Beatriz",
            }),
            sample({
                "osnet_x1_0": "Ana", "repvit": "Ana",
                "efficientformerv2": "Desconocido", "fastvit": "Beatriz",
                "swiftformer": "Beatriz",
            }),
            sample({
                "osnet_x1_0": "Ana", "repvit": "Ana",
                "efficientformerv2": "Desconocido", "fastvit": "Beatriz",
                "swiftformer": "Beatriz",
            }),
            sample({
                "osnet_x1_0": "Ana", "repvit": "Ana",
                "efficientformerv2": "Desconocido", "fastvit": "Beatriz",
                "swiftformer": "Beatriz",
            }),
        ]

        identity, confidence, votes = resolve_temporal_consensus(samples)

        self.assertEqual(identity, "Desconocido")
        self.assertEqual(confidence, 0.0)
        self.assertEqual(votes["osnet_x1_0"], "Ana")
        self.assertEqual(votes["fastvit"], "Beatriz")

    def test_requires_at_least_three_architectures_to_agree(self):
        result = sample({
            "osnet_x1_0": "Ana", "repvit": "Ana",
            "efficientformerv2": "Beatriz", "fastvit": "Desconocido",
            "swiftformer": "Desconocido",
        })

        identity, confidence, _ = resolve_temporal_consensus([result])

        self.assertEqual(identity, "Desconocido")
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
