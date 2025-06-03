"""
RewardAnything: Generalizable Principle-Following Reward Models

This package provides both local and remote inference capabilities for 
RewardAnything models that can follow natural language evaluation principles.
"""

from .local import from_pretrained
from .client import Client
from .models import RewardResult, RewardRequest, RewardResponse
# from .benchmarks import RABench

__version__ = "1.0.0"
__all__ = ["from_pretrained", "Client", "RewardResult", "RewardRequest", "RewardResponse"]

# Optional benchmarks import (only if available)
try:
    from .benchmarks import RABench
    __all__.append("RABench")
except ImportError:
    pass
