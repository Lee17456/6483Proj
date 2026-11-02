"""Small mathematical and gradient smoke tests for the new methods."""

import unittest

import torch

from advanced_longtail import ICLResNet18, ICLState, RDRState, icl_loss


class AdvancedLongtailTests(unittest.TestCase):
    def test_rdr_weights_are_positive_normalized_and_respond_to_counts(self):
        features = torch.eye(4, 512)
        labels = torch.tensor([0, 0, 1, 1])
        state = RDRState(torch.tensor([10, 2]), 512, momentum=0.9)
        state.update(features, labels)
        weights = state.weights(features, labels, ridge=1e-4, temperature=1.0)
        self.assertTrue(bool(torch.isfinite(weights).all()))
        self.assertTrue(bool((weights > 0).all()))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        self.assertGreater(float(weights[2:].mean()), float(weights[:2].mean()))

    def test_icl_all_experts_receive_gradients(self):
        model = ICLResNet18(classes=10, experts=2)
        model.train()
        logits, features = model.expert_forward(torch.randn(2, 3, 32, 32))
        labels = torch.tensor([3, 5])
        state = ICLState(2, 10, 512, torch.device("cpu"), momentum=0.1)
        args = type("Settings", (), {"label_smoothing": 0.1, "icl_ad_until": 1,
                                      "icl_ccl_after": 5, "icl_topk": 3,
                                      "icl_temperature": 0.1, "icl_kd_weight": 1.0,
                                      "icl_ad_weight": 0.05, "icl_ccl_weight": 0.1})()
        loss, _ = icl_loss(logits, features, labels, state, epoch=1, args=args)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for classifier in model.classifiers:
            self.assertIsNotNone(classifier.weight.grad)
            self.assertGreater(float(classifier.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
