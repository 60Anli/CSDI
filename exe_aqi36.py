import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import yaml

from dataset_aqi36 import get_dataloader
from main_model import CSDI_AQI36
from utils import evaluate, train


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    with open(os.path.join("config", args.config), "r") as f:
        config = yaml.safe_load(f)

    target_strategy = args.targetstrategy
    if target_strategy == "hybrid":
        target_strategy = "mix"
    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = target_strategy
    config["seed"] = args.seed
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.itr_per_epoch is not None:
        config["train"]["itr_per_epoch"] = args.itr_per_epoch

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f"./save/aqi36_raap_{current_time}/"
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    mask_sensor = config["model"].get("mask_sensor", [])
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"],
        device=args.device,
        val_len=args.val_len,
        num_workers=args.num_workers,
        seed=args.seed,
        mask_sensor=mask_sensor,
    )
    model = CSDI_AQI36(config, args.device).to(args.device)

    if model.raap is not None:
        bank_count = model.build_raap_bank(train_loader)
        print(f"RAAP retrieval bank built with {bank_count} items")

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            valid_epoch_interval=config["train"].get("valid_epoch_interval", 20),
            foldername=foldername,
            resume_path=args.resume,
        )
    else:
        model.load_state_dict(
            torch.load(
                os.path.join("./save", args.modelfolder, "model.pth"),
                map_location=args.device,
            )
        )

    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSDI AQI36 with optional RAAP")
    parser.add_argument("--config", type=str, default="base_aqi36_raap.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy",
        type=str,
        default="mix",
        choices=["mix", "random", "historical", "hybrid"],
    )
    parser.add_argument("--val_len", type=float, default=0.1)
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--itr_per_epoch", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)

    main(parser.parse_args())
