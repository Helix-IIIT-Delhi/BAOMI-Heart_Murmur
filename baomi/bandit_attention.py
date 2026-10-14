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
        seq_len = codec_feat.size(1)

        # Project and reshape for multi-head attention
        Q = self.query(codec_feat).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(spectral_feat).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(spectral_feat).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim] for proper attention
        Q = Q.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Compute scaled dot-product attention per head
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch, num_heads, seq_len, head_dim]

        # Compute rewards as mean attention weight per head
        rewards = attention_weights.mean(dim=(2, 3))  # [batch, num_heads]
        rewards = rewards.mean(dim=0)  # [num_heads]

        # Update Q-values with bandit algorithm
        new_q = self.q_values * self.decay + rewards * (1 - self.decay)
        self.q_values.data = new_q

        # Get head weights based on Q-values
        head_weights = F.softmax(self.q_values, dim=0)  # [num_heads]

        # Weighted combination of heads: [batch, num_heads, seq_len, head_dim]
        weighted_attended = attended * head_weights.view(1, -1, 1, 1)
        output = weighted_attended.sum(dim=1)  # [batch, seq_len, head_dim]

        # Reshape back to [batch, seq_len, d_model]
        output = output.reshape(batch_size, seq_len, self.d_model)

        return self.output(output)
