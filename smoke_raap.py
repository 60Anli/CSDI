import copy

import torch

from main_model import CSDI_AQI36
from raap import RAAPModule


def make_config(raap_enabled, fusion="internal"):
    return {
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 1.0e-3,
            "itr_per_epoch": 1,
        },
        "diffusion": {
            "layers": 1,
            "channels": 8,
            "nheads": 2,
            "diffusion_embedding_dim": 16,
            "beta_start": 0.0001,
            "beta_end": 0.2,
            "num_steps": 4,
            "schedule": "quad",
            "is_linear": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 8,
            "featureemb": 4,
            "target_strategy": "random",
            "mask_sensor": [],
        },
        "raap": {
            "enabled": raap_enabled,
            "fusion": fusion,
            "top_k": 2,
            "bank_size": 8,
            "patch_size": 5,
            "d_model": 16,
            "n_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dropout": 0.0,
            "retrieval_value_weight": 1.0,
            "retrieval_mask_weight": 0.1,
            "retrieval_distance": "rmse",
            "retrieval_score_chunk_size": 4,
            "retrieval_no_overlap_penalty": 3.0,
            "fusion_dropout": 0.0,
            "fusion_gate_init": -2.0,
            "use_batch_retrieval_fallback": True,
        },
    }


def make_batch(batch_size=2, length=37, target_dim=36):
    observed_data = torch.randn(batch_size, length, target_dim)
    observed_mask = (torch.rand(batch_size, length, target_dim) > 0.1).float()
    gt_mask = observed_mask.clone()
    gt_mask[:, ::3, :] = 0
    hist_mask = observed_mask.clone()
    hist_mask[:, 1::4, :] = 0
    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "gt_mask": gt_mask,
        "hist_mask": hist_mask,
        "timepoints": torch.arange(length).float().unsqueeze(0).repeat(batch_size, 1),
        "cut_length": torch.zeros(batch_size).long(),
    }


def test_raap_shape():
    config = make_config(True)["raap"]
    raap = RAAPModule(config, target_dim=36)
    observed_data = torch.randn(3, 36, 37)
    cond_mask = torch.randint(0, 2, (3, 36, 37)).float()
    x_rag_prior = raap(observed_data, cond_mask)
    _, reference_values, reference_masks = raap(observed_data, cond_mask, return_references=True)
    assert x_rag_prior.shape == observed_data.shape
    assert reference_values.shape == (3, 2, 36, 37)
    assert reference_masks.shape == (3, 2, 36, 37)

    bank_values = torch.randn(4, 36, 37)
    bank_masks = torch.randint(0, 2, (4, 36, 37)).float()
    with torch.no_grad():
        bank_embeds = raap.encode_global(bank_values, bank_masks)
    raap.retrieval_bank.set_bank(bank_values, bank_masks, bank_embeds)
    x_rag_prior, reference_values, reference_masks = raap(
        observed_data, cond_mask, return_references=True
    )
    assert x_rag_prior.shape == observed_data.shape
    assert reference_values.shape == (3, 2, 36, 37)
    assert reference_masks.shape == (3, 2, 36, 37)


def test_csdi_forward_backward(raap_enabled, fusion="internal"):
    config = make_config(raap_enabled, fusion=fusion)
    model = CSDI_AQI36(copy.deepcopy(config), "cpu")
    batch = make_batch()
    loss = model(batch, is_train=1)
    loss.backward()
    assert torch.isfinite(loss).item()


if __name__ == "__main__":
    test_raap_shape()
    test_csdi_forward_backward(False)
    test_csdi_forward_backward(True, fusion="input")
    test_csdi_forward_backward(True, fusion="internal")
    print("RAAP smoke tests passed")
