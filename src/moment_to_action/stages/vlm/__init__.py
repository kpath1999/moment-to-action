"""VLM pipeline stages — vision-language model classification and video description."""

from __future__ import annotations

from ._mobileclip import MobileCLIPStage
from ._oracle_dino import OracleGroundingDinoStage
from ._oracle_siglip import OracleSigLipStage
from ._smolvlm2 import SmolVLM2Stage

__all__ = ["MobileCLIPStage", "OracleGroundingDinoStage", "OracleSigLipStage", "SmolVLM2Stage"]
