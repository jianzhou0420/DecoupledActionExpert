"""Composed Adaptor that bridges RobotAdaptor and ModelAdaptor.

The Adaptor class composes a RobotAdaptor (robot-specific) and a ModelAdaptor
(model-specific) via the canonical intermediate format. This enables any
(robot, model) combination to work without writing new adaptor classes.

Data flow:
    Dataset -> Robot.dataset_to_canonical() -> Model.canonical_to_model() -> Model
    Env     -> Robot.env_to_canonical()     -> Model.canonical_to_model() -> Model
    Model   -> Model.model_to_canonical()   -> Robot.canonical_to_env()   -> Env
"""

from __future__ import annotations

import logging

from vlaworkspace.adaptors.models.base_model import ModelAdaptor
from vlaworkspace.adaptors.robots.base_robot import RobotAdaptor

logger = logging.getLogger(__name__)


class Adaptor:
    """Composed adaptor bridging Robot + Model via canonical format.

    Delegates robot-specific transforms to a RobotAdaptor and model-specific
    transforms to a ModelAdaptor, connected via a canonical intermediate format.

    Args:
        robot: RobotAdaptor instance (handles robot-specific conversions).
        model: ModelAdaptor instance (handles model-specific conversions).
    """

    def __init__(
        self,
        *,
        robot: RobotAdaptor,
        model: ModelAdaptor,
    ) -> None:
        self.robot = robot
        self.model = model
        self._norm_stats = model.get_norm_stats()

        logger.info(
            f"Adaptor created: robot={type(robot).__name__}, model={type(model).__name__}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Train / eval mode
    # ─────────────────────────────────────────────────────────────────────

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def datasets_input_transforms(self, data: dict) -> dict:
        """Transform LeRobot dataset sample -> model input.

        Pipeline: Robot.dataset_to_canonical() -> Model.canonical_to_model()
        """
        canonical = self.robot.dataset_to_canonical(data)
        return self.model.canonical_to_model(canonical)

    def env_input_transforms(self, data: dict) -> dict:
        """Transform robot env observation -> model input.

        Pipeline: Robot.env_to_canonical() -> Model.canonical_to_model()
        """
        canonical = self.robot.env_to_canonical(data)
        return self.model.canonical_to_model(canonical)

    def output_transforms(self, data: dict) -> dict:
        """Transform model output -> robot env actions.

        Pipeline: Model.model_to_canonical() -> Robot.canonical_to_env()
        """
        info = self.robot.get_canonical_info()
        canonical_action = self.model.model_to_canonical(data, info)

        # Extract state from data if present (needed for delta->absolute)
        state = None
        if "state" in data:
            state = data

        return self.robot.canonical_to_env(canonical_action, state=state)

    # ─────────────────────────────────────────────────────────────────────
    # Dimensions (delegate to robot)
    # ─────────────────────────────────────────────────────────────────────

    def get_state_dim(self) -> int:
        return self.robot.get_state_dim()

    def get_action_dim(self) -> int:
        return self.robot.get_action_dim()

    # ─────────────────────────────────────────────────────────────────────
    # Norm stats (delegate to model)
    # ─────────────────────────────────────────────────────────────────────

    def get_norm_stats(self) -> dict | None:
        return self.model.get_norm_stats()

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return self.model.get_norm_stats_keys()

    def get_norm_stats_mode(self) -> str:
        return self.model.get_norm_stats_mode()
