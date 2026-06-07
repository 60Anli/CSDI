import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MissingAwarePatchEmbed(nn.Module):
    def __init__(self, patch_size, d_model, target_dim, max_patches=512, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.target_dim = target_dim
        self.proj = nn.Linear(2 * patch_size, d_model)
        self.variable_embedding = nn.Embedding(target_dim, d_model)
        self.position_embedding = nn.Embedding(max_patches, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observed_data, cond_mask):
        B, K, L = observed_data.shape
        P = self.patch_size
        n_patches = math.ceil(L / P)
        padded_len = n_patches * P
        pad_len = padded_len - L

        known_values = observed_data * cond_mask
        x = torch.stack([known_values, cond_mask], dim=-1)
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.reshape(B, K, n_patches, P * 2)
        tokens = self.proj(x).reshape(B, K * n_patches, -1)

        var_ids = torch.arange(K, device=observed_data.device)
        var_ids = var_ids.clamp(max=self.target_dim - 1)
        patch_ids = torch.arange(n_patches, device=observed_data.device)
        patch_ids = patch_ids.clamp(max=self.position_embedding.num_embeddings - 1)

        var_emb = self.variable_embedding(var_ids).unsqueeze(1).expand(K, n_patches, -1)
        pos_emb = self.position_embedding(patch_ids).unsqueeze(0).expand(K, -1, -1)
        tokens = tokens + (var_emb + pos_emb).reshape(1, K * n_patches, -1)

        meta = {
            "K": K,
            "L": L,
            "n_patches": n_patches,
            "padded_len": padded_len,
        }
        return self.dropout(tokens), meta


class AttentionEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        encoded = self.encoder(tokens)
        query = self.pool_query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, need_weights=False)
        return encoded, self.norm(pooled.squeeze(1))


class RetrievalBank(nn.Module):
    def __init__(self, top_k=5, bank_size=2048):
        super().__init__()
        self.top_k = top_k
        self.bank_size = bank_size
        self.register_buffer("values", torch.empty(0), persistent=False)
        self.register_buffer("masks", torch.empty(0), persistent=False)
        self.register_buffer("embeds", torch.empty(0), persistent=False)

    @property
    def is_ready(self):
        return self.values.numel() > 0 and self.embeds.numel() > 0

    def set_bank(self, values, masks, embeds):
        size = min(values.shape[0], self.bank_size)
        self.values = values[:size].detach()
        self.masks = masks[:size].detach()
        self.embeds = F.normalize(embeds[:size].detach(), dim=-1)

    def retrieve(self, query_embed, dtype):
        if not self.is_ready:
            return None, None

        bank_embed = self.embeds.to(device=query_embed.device, dtype=query_embed.dtype)
        bank_values = self.values.to(device=query_embed.device, dtype=dtype)
        bank_masks = self.masks.to(device=query_embed.device, dtype=dtype)

        query_embed = F.normalize(query_embed, dim=-1)
        sim = torch.matmul(query_embed, bank_embed.transpose(0, 1))
        k_eff = min(self.top_k, bank_embed.shape[0])
        top_idx = sim.topk(k_eff, dim=-1).indices

        retrieved_values = bank_values[top_idx]
        retrieved_masks = bank_masks[top_idx]
        if k_eff < self.top_k:
            retrieved_values, retrieved_masks = self._pad_retrieval(
                retrieved_values, retrieved_masks, self.top_k - k_eff
            )
        return retrieved_values, retrieved_masks

    @staticmethod
    def _pad_retrieval(values, masks, pad_count):
        pad_shape = list(values.shape)
        pad_shape[1] = pad_count
        values_pad = values.new_zeros(pad_shape)
        masks_pad = masks.new_zeros(pad_shape)
        return torch.cat([values, values_pad], dim=1), torch.cat([masks, masks_pad], dim=1)


class RetrievalCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens, memory_tokens):
        attended, _ = self.cross_attn(
            query_tokens, memory_tokens, memory_tokens, need_weights=False
        )
        h = self.norm1(query_tokens + self.dropout(attended))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class AttentionPriorDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, patch_size, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.patch_reconstruct = nn.Linear(d_model, patch_size)
        self.patch_size = patch_size

    def forward(self, rag_tokens, meta):
        B = rag_tokens.shape[0]
        K = meta["K"]
        L = meta["L"]
        n_patches = meta["n_patches"]

        decoded = self.decoder(rag_tokens)
        patches = self.patch_reconstruct(decoded)
        prior = patches.reshape(B, K, n_patches, self.patch_size).reshape(
            B, K, n_patches * self.patch_size
        )
        return prior[..., :L]


class RAAPModule(nn.Module):
    def __init__(self, config, target_dim):
        super().__init__()
        self.enabled = config.get("enabled", False)
        self.top_k = config.get("top_k", 5)
        self.bank_size = config.get("bank_size", 2048)
        self.patch_size = config.get("patch_size", 4)
        self.d_model = config.get("d_model", 128)
        self.n_heads = config.get("n_heads", 4)
        self.use_batch_retrieval_fallback = config.get(
            "use_batch_retrieval_fallback", True
        )

        dropout = config.get("dropout", 0.1)
        self.patch_embed = MissingAwarePatchEmbed(
            patch_size=self.patch_size,
            d_model=self.d_model,
            target_dim=target_dim,
            max_patches=config.get("max_patches", 512),
            dropout=dropout,
        )
        self.encoder = AttentionEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=config.get("num_encoder_layers", 2),
            dropout=dropout,
        )
        self.retrieval_bank = RetrievalBank(top_k=self.top_k, bank_size=self.bank_size)
        self.cross_attention = RetrievalCrossAttention(
            d_model=self.d_model, n_heads=self.n_heads, dropout=dropout
        )
        self.decoder = AttentionPriorDecoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=config.get("num_decoder_layers", 1),
            patch_size=self.patch_size,
            dropout=dropout,
        )

    def encode_global(self, observed_data, cond_mask):
        tokens, _ = self.patch_embed(observed_data, cond_mask)
        _, global_embed = self.encoder(tokens)
        return global_embed

    def forward(self, observed_data, cond_mask):
        dtype = observed_data.dtype
        cond_mask = cond_mask.to(dtype=dtype, device=observed_data.device)

        query_tokens, meta = self.patch_embed(observed_data, cond_mask)
        query_tokens, query_global = self.encoder(query_tokens)

        retrieved_values, retrieved_masks = self.retrieval_bank.retrieve(
            query_global, dtype=dtype
        )
        if retrieved_values is None and self.use_batch_retrieval_fallback:
            retrieved_values, retrieved_masks = self._batch_retrieval(
                observed_data, cond_mask, query_global
            )

        if retrieved_values is None:
            memory_tokens = query_tokens.new_zeros(query_tokens.shape)
        else:
            memory_tokens = self._encode_retrieved_segments(
                retrieved_values, retrieved_masks
            )

        rag_tokens = self.cross_attention(query_tokens, memory_tokens)
        return self.decoder(rag_tokens, meta).to(dtype=dtype)

    def _encode_retrieved_segments(self, values, masks):
        B, top_k, K, L = values.shape
        flat_values = values.reshape(B * top_k, K, L)
        flat_masks = masks.reshape(B * top_k, K, L)
        tokens, _ = self.patch_embed(flat_values, flat_masks)
        retr_tokens, _ = self.encoder(tokens)
        return retr_tokens.reshape(B, top_k * retr_tokens.shape[1], -1)

    def _batch_retrieval(self, observed_data, cond_mask, query_global):
        B, K, L = observed_data.shape
        if B <= 1 or self.top_k <= 0:
            return None, None

        query = F.normalize(query_global, dim=-1)
        sim = torch.matmul(query, query.transpose(0, 1))
        sim.fill_diagonal_(-torch.inf)
        k_eff = min(self.top_k, B - 1)
        top_idx = sim.topk(k_eff, dim=-1).indices

        values = observed_data.detach()[top_idx]
        masks = cond_mask.detach()[top_idx]
        if k_eff < self.top_k:
            values, masks = RetrievalBank._pad_retrieval(
                values, masks, self.top_k - k_eff
            )
        return values, masks
