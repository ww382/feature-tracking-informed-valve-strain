#!/usr/bin/env python3
"""
Distance-weighted deformable CPD registration.

Extends pycpd.DeformableRegistration with an exponential distance-decay
term in the E-step so that distant source points contribute less to the
correspondence probabilities.
"""

import numpy as np
from pycpd import DeformableRegistration


class DistanceWeightedDeformableCPD(DeformableRegistration):
    def __init__(self, decay_factor=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_factor = decay_factor  # Controls how quickly influence diminishes

    def expectation(self):
        """Override expectation step to downscale contributions from distant source points."""
        # Gaussian similarity matrix (M, N)
        P = np.exp(
            -np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)
            / (2 * self.sigma2)
        )

        # Outlier term
        c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1.0 - self.w) * self.M / self.N
        den = np.clip(np.sum(P, axis=0, keepdims=True), np.finfo(self.X.dtype).eps, None) + c
        self.P = P / den

        # Apply exponential distance-decay and re-normalise
        distances = np.linalg.norm(self.X[:, None, :] - self.TY[None, :, :], axis=2)
        self.P *= np.exp(-self.decay_factor * distances)
        self.P /= np.sum(self.P, axis=0, keepdims=True)

        # Sufficient statistics
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)