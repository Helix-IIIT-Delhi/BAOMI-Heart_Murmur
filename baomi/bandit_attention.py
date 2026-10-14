"""Bandit-based Cross-Attention Mechanism"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BanditCrossAttention(nn.Module):
    """
    Bandit-based Multi-Head Cross-Attention Layer

    Uses multi-armed bandit algorithm to dynamically weight attention heads,
    learning which heads capture murmur-indicative information.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.decay = decay
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.q_values = nn.Parameter(torch.ones(num_heads), requires_grad=False)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, codec_feat, spectral_feat):
        """Multi-head cross-attention with bandit head selection"""
        batch_size = codec_feat.size(0)

        Q = self.query(codec_feat).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.key(spectral_feat).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.value(spectral_feat).view(batch_size, -1, self.num_heads, self.head_dim)

        # Compute attention per head
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute rewards (mean attention weight per head as reward signal)
        rewards = attention_weights.mean(dim=(0, 1, 2))

        # Update Q-values with bandit algorithm
        new_q = self.q_values * self.decay + rewards * (1 - self.decay)
        self.q_values.data = new_q

        # Get best heads and apply weighted combination
        head_weights = F.softmax(self.q_values, dim=0)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch, seq, num_heads, head_dim]

        # Weighted combination of heads
        output = (attended * head_weights.view(1, 1, -1, 1)).sum(dim=2)

        return self.output(output)
