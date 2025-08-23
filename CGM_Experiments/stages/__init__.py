"""
CGM Stage implementations

This module contains the four stages of the Common Governance Model:
- CS (Common Source)
- UNA (Unity Non-Absolute)
- ONA (Opposition Non-Absolute)
- BU (Balance Universal)
"""

from .cs_stage import CSStage
from .una_stage import UNAStage
from .ona_stage import ONAStage
from .bu_stage import BUStage

__all__ = ["CSStage", "UNAStage", "ONAStage", "BUStage"]
