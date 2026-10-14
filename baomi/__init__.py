"""BAOMI: Bandit-based Audio Cross-Attention for Heart Murmur Classification"""

__version__ = "1.0.0"

from .model import BAOMIModel
from .bandit_attention import BanditCrossAttention

__all__ = ["BAOMIModel", "BanditCrossAttention"]
