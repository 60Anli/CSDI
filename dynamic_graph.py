import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphEncoder(nn.Module):
    def __init__(self, graph_dim=32, graph_topk=8, temperature=0.1):
        super().__init__()
        self.graph_topk = graph_topk
        self.temperature = temperature
        self.feature_mlp = nn.Sequential(
            nn.Linear(4, graph_dim),
            nn.ReLU(),
            nn.Linear(graph_dim, graph_dim),
        )
        self.message_proj = nn.Linear(graph_dim, graph_dim)

    def forward(self, observed_data, cond_mask):
        # observed_data, cond_mask: (B,K,L)
        B, K, L = observed_data.shape
        mask = cond_mask.float()
        count = mask.sum(dim=-1)
        denom = count.clamp_min(1.0)

        masked_data = observed_data * mask
        mean = masked_data.sum(dim=-1) / denom
        centered = (observed_data - mean.unsqueeze(-1)) * mask
        std = torch.sqrt((centered.pow(2).sum(dim=-1) / denom).clamp_min(1e-6))

        time_index = torch.arange(L, device=observed_data.device).view(1, 1, L)
        last_index = (mask * time_index).long().max(dim=-1).values
        last_value = observed_data.gather(-1, last_index.unsqueeze(-1)).squeeze(-1)
        last_value = torch.where(count > 0, last_value, torch.zeros_like(last_value))

        missing_ratio = 1.0 - count / max(L, 1)
        stats = torch.stack([mean, std, last_value, missing_ratio], dim=-1)

        H = self.feature_mlp(stats)
        if K == 1:
            return H + self.message_proj(torch.zeros_like(H))

        affinity = torch.matmul(F.normalize(H, dim=-1), F.normalize(H, dim=-1).transpose(1, 2))
        affinity = affinity / max(self.temperature, 1e-6)
        affinity = affinity.masked_fill(
            torch.eye(K, device=observed_data.device, dtype=torch.bool).unsqueeze(0),
            float("-inf"),
        )

        topk = min(self.graph_topk, K - 1)
        if topk < 1:
            return H + self.message_proj(torch.zeros_like(H))
        topk_index = affinity.topk(topk, dim=-1).indices
        topk_mask = torch.zeros_like(affinity, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_index, True)
        affinity = affinity.masked_fill(~topk_mask, float("-inf"))
        graph_weight = torch.softmax(affinity, dim=-1)

        message = torch.matmul(graph_weight, H)
        return self.message_proj(message) + H
