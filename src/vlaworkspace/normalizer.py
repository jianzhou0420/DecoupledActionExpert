"""Standalone normalizer with streaming stats and two normalization modes.

Supports:
- **minmax**: maps to [-1, 1] — matches ``DPModel._normalize_limits``
- **zscore**: (x - mean) / (std + eps) — matches ``Pi0Model._normalize``

Usage::

    # Streaming (for large datasets)
    norm = Normalizer(dim=10)
    for batch in loader:
        norm.update(batch)
    stats = norm.finalize()

    # All-at-once
    norm = Normalizer.from_data(arr)            # [N, dim]

    # From pre-computed constants (e.g. dp_defaults.py)
    norm = Normalizer.from_dict(stats_dict)

    # Normalize / denormalize
    y = norm.normalize(x, mode="minmax")
    x_hat = norm.denormalize(y, mode="minmax")
"""

from __future__ import annotations

import numpy as np

# Epsilon constants — match existing model code exactly.
_MINMAX_EPS = 1e-7   # dp_model.py:480 — range guard
_ZSCORE_EPS = 1e-6   # pi0_model.py:350 — additive std guard


class Normalizer:
    """Reusable normalizer with streaming statistics computation."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, dim: int) -> None:
        """Create an empty normalizer for streaming updates.

        Parameters
        ----------
        dim : int
            Dimensionality of each sample (last axis).
        """
        self._dim = dim
        self._count: int = 0
        self._min_val = np.full(dim, np.inf, dtype=np.float64)
        self._max_val = np.full(dim, -np.inf, dtype=np.float64)
        self._sum_val = np.zeros(dim, dtype=np.float64)
        self._sum_sq = np.zeros(dim, dtype=np.float64)
        self._stats: dict[str, np.ndarray] | None = None

    @classmethod
    def from_data(cls, arr: np.ndarray) -> Normalizer:
        """Create a finalized normalizer from a complete data array.

        Parameters
        ----------
        arr : np.ndarray
            Array of shape ``[N, dim]`` (or ``[..., dim]``).
        """
        arr = np.asarray(arr)
        dim = arr.shape[-1]
        norm = cls(dim)
        norm.update(arr)
        norm.finalize()
        return norm

    @classmethod
    def from_dict(cls, stats_dict: dict[str, np.ndarray | list]) -> Normalizer:
        """Create a finalized normalizer from a pre-computed stats dict.

        The dict must contain at least ``"min"`` and ``"max"`` keys.
        ``"mean"`` and ``"std"`` are optional (required only for zscore mode).

        Parameters
        ----------
        stats_dict : dict
            Keys: ``"min"``, ``"max"``, and optionally ``"mean"``, ``"std"``.
        """
        stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats_dict.items()
                 if k in ("min", "max", "mean", "std")}
        if "min" not in stats or "max" not in stats:
            raise ValueError("stats_dict must contain 'min' and 'max' keys")
        dim = stats["min"].shape[-1]
        norm = cls(dim)
        norm._stats = stats
        return norm

    # ------------------------------------------------------------------
    # Streaming updates
    # ------------------------------------------------------------------

    def update(self, batch: np.ndarray) -> None:
        """Accumulate statistics from a batch.

        Parameters
        ----------
        batch : np.ndarray
            Array of shape ``[B, dim]`` or ``[..., dim]``.

        Raises
        ------
        RuntimeError
            If called after :meth:`finalize`.
        """
        if self._stats is not None:
            raise RuntimeError("Cannot update after finalize()")

        batch = np.asarray(batch, dtype=np.float64)
        flat = batch.reshape(-1, self._dim)
        n = flat.shape[0]

        self._count += n
        self._min_val = np.minimum(self._min_val, flat.min(axis=0))
        self._max_val = np.maximum(self._max_val, flat.max(axis=0))
        self._sum_val += flat.sum(axis=0)
        self._sum_sq += (flat ** 2).sum(axis=0)

    def finalize(self) -> dict[str, np.ndarray]:
        """Compute final statistics from accumulated data.

        Returns
        -------
        dict
            ``{"min": ..., "max": ..., "mean": ..., "std": ...}`` as float32.

        Raises
        ------
        RuntimeError
            If no samples have been accumulated.
        """
        if self._count == 0:
            raise RuntimeError("Cannot finalize with 0 samples")

        mean = self._sum_val / self._count
        var = self._sum_sq / self._count - mean ** 2
        var = np.maximum(var, 0.0)  # guard against negative from float rounding
        std = np.sqrt(var)

        self._stats = {
            "min": self._min_val.astype(np.float32),
            "max": self._max_val.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Normalize / denormalize
    # ------------------------------------------------------------------

    def normalize(self, x: np.ndarray, mode: str = "minmax") -> np.ndarray:
        """Normalize data.

        Parameters
        ----------
        x : np.ndarray
            Input data with last dimension matching ``dim``.
        mode : str
            ``"minmax"`` (→ [-1, 1]) or ``"zscore"`` (→ zero-mean unit-var).
        """
        self._check_finalized()
        x = np.asarray(x, dtype=np.float32)
        if mode == "minmax":
            return self._normalize_minmax(x)
        elif mode == "zscore":
            return self._normalize_zscore(x)
        else:
            raise ValueError(f"Unknown mode {mode!r}, expected 'minmax' or 'zscore'")

    def denormalize(self, y: np.ndarray, mode: str = "minmax") -> np.ndarray:
        """Denormalize data (inverse of :meth:`normalize`).

        Parameters
        ----------
        y : np.ndarray
            Normalized data.
        mode : str
            Must match the mode used for normalization.
        """
        self._check_finalized()
        y = np.asarray(y, dtype=np.float32)
        if mode == "minmax":
            return self._denormalize_minmax(y)
        elif mode == "zscore":
            return self._denormalize_zscore(y)
        else:
            raise ValueError(f"Unknown mode {mode!r}, expected 'minmax' or 'zscore'")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return stats as a dict of float32 arrays."""
        self._check_finalized()
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_finalized(self) -> None:
        if self._stats is None:
            raise RuntimeError("Normalizer has not been finalized yet")

    def _normalize_minmax(self, x: np.ndarray) -> np.ndarray:
        """Range normalization: min/max → [-1, 1].  Matches DPModel._normalize_limits."""
        min_val = self._stats["min"]
        max_val = self._stats["max"]
        input_range = max_val - min_val
        input_range = np.where(input_range < _MINMAX_EPS, 2.0, input_range)
        scale = 2.0 / input_range
        offset = -1.0 - scale * min_val
        return (x * scale + offset).astype(np.float32)

    def _denormalize_minmax(self, y: np.ndarray) -> np.ndarray:
        """Inverse of minmax normalization.  Matches DPModel._unnormalize_limits."""
        min_val = self._stats["min"]
        max_val = self._stats["max"]
        input_range = max_val - min_val
        input_range = np.where(input_range < _MINMAX_EPS, 2.0, input_range)
        scale = 2.0 / input_range
        offset = -1.0 - scale * min_val
        return ((y - offset) / scale).astype(np.float32)

    def _normalize_zscore(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalization.  Matches Pi0Model._normalize."""
        mean = self._stats["mean"]
        std = self._stats["std"]
        return ((x - mean) / (std + _ZSCORE_EPS)).astype(np.float32)

    def _denormalize_zscore(self, y: np.ndarray) -> np.ndarray:
        """Inverse of z-score normalization."""
        mean = self._stats["mean"]
        std = self._stats["std"]
        return (y * (std + _ZSCORE_EPS) + mean).astype(np.float32)
