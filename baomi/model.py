"""BAOMI Model for Heart Murmur Classification"""

import torch
import torch.nn as nn
from .bandit_attention import BanditCrossAttention


class BAOMIModel(nn.Module):
    """
    BAOMI: Fusion of Codec and Spectral Features via Bandit Cross-Attention

    Args:
        codec_dim: Codec representation dimension
        spectral_dim: Spectral feature dimension
        num_classes: Number of output classes
        num_heads: Number of attention heads
    """

    def __init__(self, codec_dim: int = 256, spectral_dim: int = 128,
                 num_classes: int = 2, num_heads: int = 8):
        super().__init__()
        self.d_model = codec_dim
        self.num_classes = num_classes

        # Feature projection
        self.codec_proj = nn.Linear(codec_dim, self.d_model)
        self.spectral_proj = nn.Linear(spectral_dim, self.d_model)

        # Bandit cross-attention
        self.bandit_attention = BanditCrossAttention(self.d_model, num_heads)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, codec_feat: torch.Tensor, spectral_feat: torch.Tensor):
        """Forward pass for heart murmur classification"""
        # Project features
        codec = self.codec_proj(codec_feat)
        spectral = self.spectral_proj(spectral_feat)

        # Apply bandit cross-attention
        fused = self.bandit_attention(codec, spectral)

        # Global pooling
        pooled = fused.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits
