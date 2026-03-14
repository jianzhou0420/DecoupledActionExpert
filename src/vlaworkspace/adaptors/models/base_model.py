"""Abstract base class for model adaptors.

A ModelAdaptor converts between canonical intermediate format and a specific
model's input/output format. It encapsulates all model-specific knowledge
(normalization, image format, tokenization, padding, rotation conversion, etc.).
"""

from __future__ import annotations

import abc
import json
import logging

import numpy as np

from vlaworkspace.adaptors.canonical import CanonicalDict, CanonicalInfo

logger = logging.getLogger(__name__)


class ModelAdaptor(abc.ABC):
    """Abstract base class for model adaptors.

    Subclasses implement the conversion between canonical format and a specific
    model's expected input/output format.

    Data flow:
        canonical obs    -> canonical_to_model()     -> model input
        model output     -> model_to_canonical()     -> canonical action
        canonical sample -> canonical_to_norm_stats_format() -> keyed for norm stats
    """

    def __init__(
        self,
        *,
        norm_stats_path: str | None = None,
        norm_stats: dict | None = None,
    ) -> None:
        self._norm_stats: dict | None = None
        self.training = True

        if norm_stats is not None:
            self._norm_stats = norm_stats
            logger.info(f"Using provided norm_stats: {list(self._norm_stats.keys())}")
        elif norm_stats_path is not None:
            self._norm_stats = self._load_norm_stats_from_path(norm_stats_path)

    @staticmethod
    def _load_norm_stats_from_path(path: str) -> dict:
        """Load normalization statistics from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Dict mapping keys to dicts with numpy arrays.
        """
        try:
            with open(path) as f:
                raw_stats = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"norm_stats file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in norm_stats file {path}: {e}")

        if "norm_stats" in raw_stats:
            raw_stats = raw_stats["norm_stats"]

        norm_stats = {}
        for key, stats in raw_stats.items():
            if isinstance(stats, dict) and "mean" in stats:
                stat_dict = {
                    "mean": np.array(stats["mean"], dtype=np.float32),
                    "std": np.array(stats["std"], dtype=np.float32),
                    "q01": np.array(stats.get("q01", stats["mean"]), dtype=np.float32),
                    "q99": np.array(stats.get("q99", stats["mean"]), dtype=np.float32),
                }
                if "min" in stats:
                    stat_dict["min"] = np.array(stats["min"], dtype=np.float32)
                if "max" in stats:
                    stat_dict["max"] = np.array(stats["max"], dtype=np.float32)
                norm_stats[key] = stat_dict

        if not norm_stats:
            raise ValueError(
                f"No valid norm_stats found in {path}. "
                f"Expected keys with 'mean' and 'std', got: {list(raw_stats.keys())}"
            )

        logger.info(f"Loaded norm_stats from {path}: {list(norm_stats.keys())}")
        return norm_stats

    def train(self):
        """Set training mode (e.g. random crop)."""
        self.training = True
        return self

    def eval(self):
        """Set evaluation mode (e.g. center crop)."""
        self.training = False
        return self

    def get_norm_stats(self) -> dict | None:
        """Get the loaded normalization statistics."""
        return self._norm_stats

    def set_norm_stats(self, norm_stats: dict) -> None:
        """Set normalization statistics programmatically."""
        self._norm_stats = norm_stats

    @abc.abstractmethod
    def canonical_to_model(self, canonical: CanonicalDict) -> dict:
        """Convert canonical observation to model input format.

        Args:
            canonical: Canonical observation dictionary.

        Returns:
            Model-specific input dictionary.
        """
        ...

    @abc.abstractmethod
    def model_to_canonical(self, model_output: dict, info: CanonicalInfo) -> CanonicalDict:
        """Convert model output to canonical action format.

        Args:
            model_output: Model-specific output dictionary.
            info: CanonicalInfo from the robot adaptor.

        Returns:
            Canonical action dictionary.
        """
        ...

    @abc.abstractmethod
    def get_norm_stats_mode(self) -> str:
        """Return normalization mode: 'gaussian' or 'limits'."""
        ...

    @abc.abstractmethod
    def get_norm_stats_keys(self) -> tuple[str, ...]:
        """Return keys used for norm stats computation in model format.

        Returns:
            Tuple of keys (e.g., ("state", "actions") for Pi0,
            or ("obs/robot0_eef_pos", "action") for DP).
        """
        ...

    @abc.abstractmethod
    def canonical_to_norm_stats_format(self, canonical: CanonicalDict) -> dict:
        """Convert canonical data to format suitable for norm stats computation.

        Maps canonical state/action components to the model's norm stats key scheme.

        Args:
            canonical: Canonical observation dictionary (with actions).

        Returns:
            Dict keyed by norm stats keys with numpy arrays.
        """
        ...

    @abc.abstractmethod
    def model_input(self) -> dict:
        """Return a description of the model input format."""
        ...

    @abc.abstractmethod
    def model_output(self) -> dict:
        """Return a description of the model output format."""
        ...
