"""
Diagnostics module for temporalcv.

Provides diagnostic tools for understanding validation results:
- Influence analysis for DM test
- Gap sensitivity analysis
"""

from temporalcv.diagnostics.influence import (
    InfluenceDiagnostic,
    compute_dm_influence,
)
from temporalcv.diagnostics.sensitivity import (
    GapSensitivityResult,
    gap_sensitivity_analysis,
)

__all__ = [
    "InfluenceDiagnostic",
    "compute_dm_influence",
    "GapSensitivityResult",
    "gap_sensitivity_analysis",
]
