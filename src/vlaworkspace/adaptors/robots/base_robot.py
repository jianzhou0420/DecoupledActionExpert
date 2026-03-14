"""Abstract base class for robot adaptors.

A RobotAdaptor converts between a specific robot environment's data format
and the canonical intermediate format. It encapsulates all robot-specific
knowledge (state layout, action format, camera configuration, etc.).
"""

from __future__ import annotations

import abc

from vlaworkspace.adaptors.canonical import CanonicalDict, CanonicalInfo


class RobotAdaptor(abc.ABC):
    """Abstract base class for robot adaptors.

    Subclasses implement the conversion between a specific robot environment's
    data format and the canonical intermediate format.

    Data flow:
        Dataset -> dataset_to_canonical() -> canonical obs
        Env     -> env_to_canonical()     -> canonical obs (no actions)
        canonical action -> canonical_to_env() -> env action
    """

    @abc.abstractmethod
    def get_canonical_info(self) -> CanonicalInfo:
        """Return metadata about this robot's canonical representation.

        Returns:
            CanonicalInfo describing state/action types and rotation representations.
        """
        ...

    @abc.abstractmethod
    def dataset_to_canonical(self, data: dict) -> CanonicalDict:
        """Convert a LeRobot dataset sample to canonical format.

        Args:
            data: Raw LeRobot dataset sample (may contain torch tensors).

        Returns:
            Canonical observation dictionary.
        """
        ...

    @abc.abstractmethod
    def env_to_canonical(self, data: dict) -> CanonicalDict:
        """Convert robot environment observation to canonical format.

        Args:
            data: Raw environment observation (numpy arrays, strings).

        Returns:
            Canonical observation dictionary (without actions).
        """
        ...

    @abc.abstractmethod
    def canonical_to_env(self, canonical_action: CanonicalDict, state: dict | None = None) -> dict:
        """Convert canonical actions to robot environment format.

        Args:
            canonical_action: Canonical action dictionary from model output.
            state: Current robot state (needed for delta->absolute conversion).

        Returns:
            Environment action dictionary (e.g., {"actions": [horizon, 7]}).
        """
        ...

    @abc.abstractmethod
    def get_state_dim(self) -> int:
        """Return the total state dimension for this robot."""
        ...

    @abc.abstractmethod
    def get_action_dim(self) -> int:
        """Return the total action dimension for this robot."""
        ...

    @abc.abstractmethod
    def get_norm_stats_keys(self) -> tuple[str, ...]:
        """Return canonical-space keys for norm stats computation.

        These keys describe what components to compute statistics over,
        using the canonical naming convention.

        Returns:
            Tuple of canonical keys (e.g., ("state/pos", "state/rot", "actions/pos")).
        """
        ...

    @abc.abstractmethod
    def env_obs(self) -> dict:
        """Return a description of the raw environment observation format."""
        ...

    @abc.abstractmethod
    def env_action(self) -> dict:
        """Return a description of the environment action format."""
        ...

    @abc.abstractmethod
    def datasets(self) -> dict:
        """Return a description of the raw LeRobot dataset format."""
        ...
