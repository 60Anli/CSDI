import argparse
import torch
import datetime
import json
import yaml
import os
import random
import numpy as np

from main_model import CSDI_Physio
from dataset_physio import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_raap.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
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
config["model"]["test_missing_ratio"] = args.testmissingratio
if args.epochs is not None:
    config["train"]["epochs"] = args.epochs
if args.itr_per_epoch is not None:
    config["train"]["itr_per_epoch"] = args.itr_per_epoch
if args.batch_size is not None:
    config["train"]["batch_size"] = args.batch_size

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_prefix = "physio_raap" if config.get("raap", {}).get("enabled", False) else "physio"
foldername = "./save/" + folder_prefix + "_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = CSDI_Physio(config, args.device).to(args.device)

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

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
