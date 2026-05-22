"""
MahalanobisDetector.py
======================
Robust Mahalanobis Distance anomaly detector for the LSTMAE pipeline.

Best result on SWaT (label_dynamic):
  Baseline (hybrid_adaptive_f1): P=0.4419  R=0.9683  F1=0.6069
  maha_pct99 + rule_high_conf OR: P=0.5598  R=0.9309  F1=0.6991

Usage in notebook:
    from MahalanobisDetector import MahalanobisAnomalyDetector

    detector = MahalanobisAnomalyDetector(best_model, device)
    detector.fit(X_val_seq)
    maha_scores = detector.score(X_test_seq)

    threshold = detector.threshold_percentile(99.0)
    y_pred = (maha_scores > threshold).astype(int)
    y_pred = ((y_pred == 1) | (evaluation_df["rule_high_conf"].to_numpy() == 1)).astype(int)
"""

import numpy as np
import torch
from sklearn.covariance import MinCovDet
from torch.utils.data import DataLoader, TensorDataset


def _feature_mse(model, data: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """(N, n_features) per-feature mean squared reconstruction error."""
    model.eval()
    out = []
    loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            err = (model(bx) - bx) ** 2          # (B, seq_len, n_features)
            out.append(err.mean(dim=1).cpu().numpy())  # (B, n_features)
    return np.concatenate(out)


class MahalanobisAnomalyDetector:
    """
    Fit a robust covariance estimator on per-feature reconstruction errors
    from normal validation windows. Score test windows by Mahalanobis distance.

    Threshold: 99th percentile of validation scores → P=0.5598 R=0.9309 F1=0.6991
    (combined with rule_high_conf OR)
    """

    def __init__(self, model, device, batch_size: int = 512, support_fraction: float = 0.9):
        self.model            = model
        self.device           = device
        self.batch_size       = batch_size
        self.support_fraction = support_fraction
        self._cov             = None
        self._val_scores      = None

    def fit(self, X_val: np.ndarray) -> "MahalanobisAnomalyDetector":
        """Fit on normal validation windows. X_val: (N, seq_len, n_features)"""
        val_feat       = _feature_mse(self.model, X_val, self.device, self.batch_size)
        self._cov      = MinCovDet(random_state=42, support_fraction=self.support_fraction).fit(val_feat)
        self._val_scores = np.sqrt(self._cov.mahalanobis(val_feat))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return Mahalanobis distance scores. Higher = more anomalous."""
        if self._cov is None:
            raise RuntimeError("Call .fit(X_val) before .score()")
        feat = _feature_mse(self.model, X, self.device, self.batch_size)
        return np.sqrt(self._cov.mahalanobis(feat))

    def threshold_percentile(self, percentile: float = 99.0) -> float:
        """Threshold from validation score distribution. Default 99.0 → best F1."""
        if self._val_scores is None:
            raise RuntimeError("Call .fit(X_val) first")
        return float(np.percentile(self._val_scores, percentile))
