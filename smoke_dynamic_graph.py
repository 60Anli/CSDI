import copy
import yaml
import torch

from dataset_physio import get_dataloader as get_physio_dataloader
from main_model import CSDI_Physio, CSDI_PM25


def clone_config():
    with open("config/base_graph.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config["diffusion"]["layers"] = 1
    config["diffusion"]["channels"] = 16
    config["diffusion"]["nheads"] = 2
    config["diffusion"]["num_steps"] = 2
    return config


def run_physio(device):
    config = clone_config()
    train_loader, _, test_loader = get_physio_dataloader(
        seed=1,
        nfold=0,
        batch_size=2,
        missing_ratio=0.1,
    )
    model = CSDI_Physio(config, device).to(device)
    model.train()
    train_batch = next(iter(train_loader))
    train_loss = model(train_batch, is_train=1)
    model.eval()
    eval_batch = next(iter(test_loader))
    samples, observed_data, target_mask, observed_mask, observed_tp = model.evaluate(
        eval_batch,
        n_samples=1,
    )
    print("Physio train loss:", float(train_loss.detach().cpu()))
    print(
        "Physio eval shapes:",
        tuple(samples.shape),
        tuple(observed_data.shape),
        tuple(target_mask.shape),
        tuple(observed_mask.shape),
        tuple(observed_tp.shape),
    )


def make_pm25_batch(batch_size=2, eval_length=36, target_dim=36):
    observed_mask = (torch.rand(batch_size, eval_length, target_dim) > 0.2).float()
    gt_mask = observed_mask * (torch.rand(batch_size, eval_length, target_dim) > 0.1).float()
    hist_mask = (torch.rand(batch_size, eval_length, target_dim) > 0.25).float()
    observed_data = torch.randn(batch_size, eval_length, target_dim) * observed_mask
    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "gt_mask": gt_mask,
        "hist_mask": hist_mask,
        "timepoints": torch.arange(eval_length).unsqueeze(0).expand(batch_size, -1),
        "cut_length": torch.zeros(batch_size).long(),
    }


def run_pm25(device):
    config = clone_config()
    config["model"]["target_strategy"] = "mix"
    model = CSDI_PM25(config, device).to(device)
    model.train()
    train_batch = make_pm25_batch()
    train_loss = model(train_batch, is_train=1)
    model.eval()
    eval_batch = make_pm25_batch()
    samples, observed_data, target_mask, observed_mask, observed_tp = model.evaluate(
        eval_batch,
        n_samples=1,
    )
    print("PM25 train loss:", float(train_loss.detach().cpu()))
    print(
        "PM25 eval shapes:",
        tuple(samples.shape),
        tuple(observed_data.shape),
        tuple(target_mask.shape),
        tuple(observed_mask.shape),
        tuple(observed_tp.shape),
    )


if __name__ == "__main__":
    torch.manual_seed(1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    run_physio(device)
    run_pm25(device)
