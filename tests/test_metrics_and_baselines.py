import numpy as np

from nadlab.baselines import ZScoreDetector, fit_threshold
from nadlab.metrics import evaluate_scores


def test_zscore_detector_separates_large_window_shift():
    rng = np.random.default_rng(3)
    normal = rng.normal(0, 0.3, size=(30, 8, 4)).astype(np.float32)
    anomalous = normal[:5].copy() + 5.0
    detector = ZScoreDetector().fit(normal)
    normal_scores = detector.score_samples(normal)
    anomaly_scores = detector.score_samples(anomalous)
    assert anomaly_scores.mean() > normal_scores.mean()


def test_evaluation_returns_core_metrics_and_predictions():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.7])
    threshold = fit_threshold(scores[labels == 0], 0.95)
    metrics, predictions = evaluate_scores(labels, scores, threshold)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] > 0
    assert len(predictions) == len(labels)
