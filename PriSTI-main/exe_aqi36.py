import argparse
import torch
import datetime
import json
import yaml
import os
import logging
import numpy as np
import random

from dataset_aqi36 import get_dataloader
from main_model import PriSTI_aqi36
from utils import train, evaluate


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    config["diffusion"]["adj_file"] = 'AQI36'
    config["seed"] = SEED
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_prefix = "pm25_raap" if config.get("raap", {}).get("enabled", False) else "pm25_outsample"
    foldername = (
        "./save/" + folder_prefix + "_" + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"], device=args.device, val_len=args.val_len,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers,
        target_strategy=args.targetstrategy, mask_sensor=config["model"]["mask_sensor"]
    )
    model = PriSTI_aqi36(config, args.device).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            resume_path=args.resume,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device))

    if model.raap is not None:
        bank_count = model.build_raap_bank(train_loader)
        print(f"RAAP retrieval bank refreshed with {bank_count} items")

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info("model_name={}".format(args.modelfolder))
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="base_raap.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=16, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
    )
    parser.add_argument(
        "--val_len", type=float, default=0.1, help="the ratio of data used for validation (value:[0-1])"
    )
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    print(args)

    main(args)
