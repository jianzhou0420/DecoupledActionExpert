"""Dataset adaptors for transforming LeRobot datasets to model-specific formats.

This module provides a modular system for adapting various robot datasets
to different model input formats. The architecture uses composable
RobotAdaptor + ModelAdaptor pairs bridged by a canonical intermediate format.

Usage:
    from vlaworkspace.adaptors import Adaptor
    from vlaworkspace.adaptors.robots import LiberoRobot
    from vlaworkspace.adaptors.models import DPModel

    adaptor = Adaptor(robot=LiberoRobot(), model=DPModel(norm_stats_path=...))
    model_input = adaptor.datasets_input_transforms(lerobot_sample)
"""

# Composable adaptor system
from vlaworkspace.adaptors.adaptor import Adaptor
from vlaworkspace.adaptors.canonical import CanonicalInfo

# Robot adaptors
from vlaworkspace.adaptors.robots import (
    DroidStage1Robot,
    LiberoRobot,
    LiberoStage1Robot,
    MimicGenRobot,
    MimicGenStage1Robot,
    RobotAdaptor,
)

# Model adaptors
from vlaworkspace.adaptors.models import (
    DPModel,
    DPStage1Model,
    ModelAdaptor,
)

__all__ = [
    "Adaptor",
    "CanonicalInfo",
    "RobotAdaptor",
    "DroidStage1Robot",
    "LiberoRobot",
    "LiberoStage1Robot",
    "MimicGenRobot",
    "MimicGenStage1Robot",
    "ModelAdaptor",
    "DPModel",
    "DPStage1Model",
]
