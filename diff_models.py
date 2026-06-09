import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ReferenceModulatedFusion(nn.Module):
    def __init__(
        self,
        channels,
        nheads,
        top_k,
        dropout=0.0,
        gate_init=-2.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.hidden_dim = 2 * channels
        attn_heads = nheads if self.hidden_dim % nheads == 0 else math.gcd(self.hidden_dim, nheads)
        attn_heads = max(attn_heads, 1)

        self.ref_projection = nn.Conv2d(2 * top_k, self.hidden_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.ref_projection.weight)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.ref_norm = nn.LayerNorm(self.hidden_dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_projection = nn.Conv2d(3 * self.hidden_dim, self.hidden_dim, kernel_size=1)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, gate_init)

    def forward(self, y, cond_info, base_shape, reference_values, reference_masks=None):
        if reference_values is None:
            return cond_info

        B, _, K, L = base_shape
        reference_values = reference_values.to(device=cond_info.device, dtype=cond_info.dtype)
        if reference_masks is None:
            reference_masks = torch.ones_like(reference_values)
        else:
            reference_masks = reference_masks.to(device=cond_info.device, dtype=cond_info.dtype)

        reference_values = self._fit_top_k(reference_values, fill_value=0.0)
        reference_masks = self._fit_top_k(reference_masks, fill_value=0.0)

        ref_input = torch.cat([reference_values * reference_masks, reference_masks], dim=1)
        ref_feat = self.ref_projection(ref_input)

        y_feat = y.reshape(B, self.hidden_dim, K, L)
        cond_feat = cond_info.reshape(B, self.hidden_dim, K, L)

        query = (y_feat + cond_feat).permute(0, 3, 2, 1).reshape(B * L, K, self.hidden_dim)
        ref = ref_feat.permute(0, 3, 2, 1).reshape(B * L, K, self.hidden_dim)
        query = self.query_norm(query)
        ref = self.ref_norm(ref)

        attended, _ = self.spatial_attn(query, ref, ref, need_weights=False)
        attended = self.out_projection(attended)
        attended = attended.reshape(B, L, K, self.hidden_dim).permute(0, 3, 2, 1)

        gate_input = torch.cat([y_feat, cond_feat, attended], dim=1)
        gate = torch.sigmoid(self.gate_projection(gate_input))
        fused = cond_feat + gate * attended
        return fused.reshape(B, self.hidden_dim, K * L)

    def _fit_top_k(self, x, fill_value):
        if x.shape[1] == self.top_k:
            return x
        if x.shape[1] > self.top_k:
            return x[:, : self.top_k]

        pad_shape = list(x.shape)
        pad_shape[1] = self.top_k - x.shape[1]
        pad = x.new_full(pad_shape, fill_value)
        return torch.cat([x, pad], dim=1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.raap_internal_fusion = config.get("raap_internal_fusion", False)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    use_reference_fusion=self.raap_internal_fusion,
                    raap_top_k=config.get("raap_top_k", 5),
                    reference_dropout=config.get("raap_fusion_dropout", 0.0),
                    reference_gate_init=config.get("raap_fusion_gate_init", -2.0),
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, raap_reference=None, raap_reference_mask=None):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(
                x,
                cond_info,
                diffusion_emb,
                raap_reference=raap_reference,
                raap_reference_mask=raap_reference_mask,
            )
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        side_dim,
        channels,
        diffusion_embedding_dim,
        nheads,
        is_linear=False,
        use_reference_fusion=False,
        raap_top_k=5,
        reference_dropout=0.0,
        reference_gate_init=-2.0,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.reference_fusion = (
            ReferenceModulatedFusion(
                channels=channels,
                nheads=nheads,
                top_k=raap_top_k,
                dropout=reference_dropout,
                gate_init=reference_gate_init,
            )
            if use_reference_fusion
            else None
        )

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, raap_reference=None, raap_reference_mask=None):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        if self.reference_fusion is not None:
            cond_info = self.reference_fusion(
                y,
                cond_info,
                base_shape,
                raap_reference,
                raap_reference_mask,
            )
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
