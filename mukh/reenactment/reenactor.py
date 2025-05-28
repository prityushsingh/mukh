"""Face reenactment module providing a unified interface for multiple reenactment models.

This module provides a factory class for creating face reenactors with different
underlying implementations. It supports multiple reenactment models through a consistent
interface.

Example:
    Basic usage with default settings:

    >>> from mukh.reenactment import FaceReenactor
    >>> reenactor = FaceReenactor.create("tps")
    >>> result = reenactor.reenact("source.jpg", "driving.jpg")

    For video reenactment:

    >>> result_path = reenactor.reenact_from_video("source.jpg", "driving.mp4")
"""

import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from .models.base_reenactor import BaseFaceReenactor
from .models.thin_plate_spline.tps_reenactor import ThinPlateSplineReenactor

ReenactorType = Literal["tps"]


class FaceReenactor:
    """Factory class for creating face reenactment model instances.

    This class provides a unified interface to create and use different face reenactment
    models through a consistent API.
    """

    @staticmethod
    def create(
        model: ReenactorType,
        model_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ) -> BaseFaceReenactor:
        """Creates a face reenactor instance of the specified type.

        Args:
            model: The type of reenactor to create. Currently supports: "tps"
                (Thin Plate Spline).
            model_path: Path to the model weights. If None, uses the default path
                for the specified model.
            device: Device to run inference on ('cpu', 'cuda'). Defaults to 'cpu'.
            **kwargs: Additional model-specific parameters.

        Returns:
            A BaseFaceReenactor instance of the requested type.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        # Define default model paths and configurations
        model_configs = {
            "tps": {
                "model_path": "mukh/reenactment/models/thin_plate_spline/vox.pth.tar",
                "config_path": "mukh/reenactment/models/thin_plate_spline/config/vox-256.yaml",
                "class": ThinPlateSplineReenactor,
            }
        }

        if model not in model_configs:
            raise ValueError(
                f"Unknown reenactor model: {model}. "
                f"Available models: {list(model_configs.keys())}"
            )

        # Use default model path if none provided
        if model_path is None:
            model_path = model_configs[model]["model_path"]

        # Create the reenactor instance
        config = model_configs[model]
        reenactor_class = config["class"]

        # For TPS model specifically
        if model == "tps":
            return reenactor_class(
                model_path=model_path,
                config_path=config["config_path"],
                device=device,
                **kwargs,
            )

        # Generic case for other models
        return reenactor_class(model_path=model_path, device=device, **kwargs)

    @staticmethod
    def list_available_models() -> List[str]:
        """Returns a list of available face reenactment model names.

        Returns:
            List of strings containing supported model names.
        """
        return ["tps"]
