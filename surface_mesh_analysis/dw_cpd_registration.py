#!/usr/bin/env python3
"""Distance-weighted deformable CPD registration."""

import numpy as np
from pycpd import DeformableRegistration


class DistanceWeightedDeformableCPD(DeformableRegistration):
    """Deformable CPD with exponential distance decay in the expectation step."""

    def __init__(self, decay_factor=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_factor = decay_factor

    def expectation(self):
        """Override expectation step to downscale contributions from distant source points."""
        P = np.exp(
            -np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)
            / (2 * self.sigma2)
        )

        c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1.0 - self.w) * self.M / self.N
        den = np.clip(np.sum(P, axis=0, keepdims=True), np.finfo(self.X.dtype).eps, None) + c
        self.P = P / den

        distances = np.linalg.norm(self.X[:, None, :] - self.TY[None, :, :], axis=2)
        self.P *= np.exp(-self.decay_factor * distances)
        self.P /= np.sum(self.P, axis=0, keepdims=True)

        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)
