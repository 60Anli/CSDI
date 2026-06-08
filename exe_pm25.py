import argparse
import torch
import datetime
import json
import yaml
import os
import random
import numpy as np

from dataset_pm25 import get_dataloader
from main_model import CSDI_PM25
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_raap.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--itr_per_epoch", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)

args = parser.parse_args()
print(args)

if args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy
if args.epochs is not None:
    config["train"]["epochs"] = args.epochs
if args.itr_per_epoch is not None:
    config["train"]["itr_per_epoch"] = args.itr_per_epoch
if args.batch_size is not None:
    config["train"]["batch_size"] = args.batch_size

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
folder_prefix = "pm25_raap" if config.get("raap", {}).get("enabled", False) else "pm25"
foldername = (
    "./save/" + folder_prefix + "_validationindex" + str(args.validationindex) + "_" + current_time + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"],
    device=args.device,
    validindex=args.validationindex,
    seed=args.seed,
)
model = CSDI_PM25(config, args.device).to(args.device)

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
        torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device)
    )

if model.raap is not None:
    bank_count = model.build_raap_bank(train_loader)
    print(f"RAAP retrieval bank refreshed with {bank_count} items")

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
