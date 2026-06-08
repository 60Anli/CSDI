import numpy as np
import sys
import torch
import torch.nn as nn
from pathlib import Path
from diff_models import Guide_diff

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from raap import RAAPModule


class PriSTI(nn.Module):
    def __init__(self, target_dim, seq_len, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.seq_len = seq_len

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_guide = config["model"]["use_guide"]
        raap_config = config.get("raap", {})
        self.raap = (
            RAAPModule(raap_config, target_dim=target_dim)
            if raap_config.get("enabled", False)
            else None
        )

        self.cde_output_channels = config["diffusion"]["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["device"] = device
        self.device = device

        if self.use_guide:
            input_dim = 2 + (1 if self.raap is not None else 0)
        else:
            input_dim = 1 if self.is_unconditional else 2
            if self.raap is not None:
                input_dim += 1
        self.diffmodel = Guide_diff(config_diff, input_dim, target_dim, self.use_guide)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def build_raap_bank(self, train_loader):
        if self.raap is None:
            return 0

        was_training = self.training
        self.eval()
        values = []
        masks = []
        embeds = []
        total = 0
        max_items = self.raap.bank_size

        with torch.no_grad():
            for batch in train_loader:
                processed = self.process_data(batch)
                observed_data = processed[0]
                observed_mask = processed[1]
                remaining = max_items - total
                if remaining <= 0:
                    break
                observed_data = observed_data[:remaining]
                observed_mask = observed_mask[:remaining]
                embed = self.raap.encode_global(observed_data, observed_mask)
                values.append(observed_data.detach())
                masks.append(observed_mask.detach())
                embeds.append(embed.detach())
                total += observed_data.shape[0]
                if total >= max_items:
                    break

        if total > 0:
            self.raap.retrieval_bank.set_bank(
                torch.cat(values, dim=0),
                torch.cat(masks, dim=0),
                torch.cat(embeds, dim=0),
            )
        if was_training:
            self.train()
        return total

    def compute_raap_prior(self, observed_data, cond_mask):
        if self.raap is None:
            return None
        x_rag_prior = self.raap(observed_data, cond_mask)
        if x_rag_prior.shape != observed_data.shape:
            raise RuntimeError(
                "RAAP prior shape must match observed_data: "
                f"{tuple(x_rag_prior.shape)} != {tuple(observed_data.shape)}"
            )
        return x_rag_prior

    def build_itp_info(self, coeffs, x_rag_prior=None):
        if not self.use_guide:
            return None

        guide_inputs = []
        if coeffs is not None:
            guide_inputs.append(coeffs.unsqueeze(1))
        elif x_rag_prior is not None:
            guide_inputs.append(torch.zeros_like(x_rag_prior).unsqueeze(1))

        if self.raap is not None:
            if x_rag_prior is None:
                if coeffs is None:
                    raise ValueError("coeffs or x_rag_prior is required for RAAP guide input")
                x_rag_prior = torch.zeros_like(coeffs)
            guide_inputs.append(x_rag_prior.to(dtype=guide_inputs[0].dtype).unsqueeze(1))

        if not guide_inputs:
            return None
        return torch.cat(guide_inputs, dim=1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, x_rag_prior=None
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data,
                cond_mask,
                observed_mask,
                side_info,
                itp_info,
                is_train,
                set_t=t,
                x_rag_prior=x_rag_prior,
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        itp_info,
        is_train,
        set_t=-1,
        x_rag_prior=None,
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(
            noisy_data, observed_data, cond_mask, x_rag_prior=x_rag_prior
        )
        if not self.use_guide:
            itp_info = cond_mask * observed_data
        predicted = self.diffmodel(total_input, side_info, t, itp_info, cond_mask)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask, x_rag_prior=None):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)
        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        if self.raap is not None and not self.use_guide:
            if x_rag_prior is None:
                x_rag_prior = torch.zeros_like(observed_data)
            x_rag_prior = x_rag_prior.to(device=total_input.device, dtype=total_input.dtype)
            total_input = torch.cat([total_input, x_rag_prior.unsqueeze(1)], dim=1)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info, x_rag_prior=None):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    if not self.use_guide:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    else:
                        diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
                if self.raap is not None and not self.use_guide:
                    if x_rag_prior is None:
                        prior_input = torch.zeros_like(observed_data)
                    else:
                        prior_input = x_rag_prior
                    prior_input = prior_input.to(device=diff_input.device, dtype=diff_input.dtype)
                    diff_input = torch.cat([diff_input, prior_input.unsqueeze(1)], dim=1)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), itp_info, cond_mask)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
        ) = self.process_data(batch)

        x_rag_prior = self.compute_raap_prior(observed_data, cond_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info = None
        if self.use_guide:
            itp_info = self.build_itp_info(coeffs, x_rag_prior)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        output = loss_func(
            observed_data,
            cond_mask,
            observed_mask,
            side_info,
            itp_info,
            is_train,
            x_rag_prior=x_rag_prior,
        )
        return output

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            x_rag_prior = self.compute_raap_prior(observed_data, cond_mask)
            side_info = self.get_side_info(observed_tp, cond_mask)
            itp_info = None
            if self.use_guide:
                itp_info = self.build_itp_info(coeffs, x_rag_prior)

            samples = self.impute(
                observed_data,
                cond_mask,
                side_info,
                n_samples,
                itp_info,
                x_rag_prior=x_rag_prior,
            )

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class PriSTI_aqi36(PriSTI):
    def __init__(self, config, device, target_dim=36, seq_len=36):
        super(PriSTI_aqi36, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )



class PriSTI_MetrLA(PriSTI):
    def __init__(self, config, device, target_dim=207, seq_len=24):
        super(PriSTI_MetrLA, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )


class PriSTI_PemsBAY(PriSTI):
    def __init__(self, config, device, target_dim=325, seq_len=24):
        super(PriSTI_PemsBAY, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )
