import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from utils.wandb import wandb_watch, wandb_log
from utils.wandb import wandb_log, wandb
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix as conf_matrix_sklearn,
    ConfusionMatrixDisplay,
    f1_score,
)

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

cpu = torch.device("cpu")
torch.manual_seed(0)
torch.cuda.list_gpu_processes()
from dataset.lungpetctdx_dataset import (
    LungPetCtDxDataset_TumorClass,
    LungPetCtDxDataset_TumorClass3D,
)
from dataset.ct_dataset import NormalizationMethods, CTDataSet

# model = torch.load("test")
from model_config.capsnet_config import Config
from model_config.resnet_config import ResnetConfig
from utils.train import train_model, EpochHistory, plot_losses, TrainHistory
from wandb.plot import confusion_matrix
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from dataset.base_dataset import BaseDataset
from typing import Literal, Any
from dataset.mnist_dataset import MNISTDataset
from eval.experiments import store_experiment


def get_model(config: dict, ds: BaseDataset):
    num_classes = len(ds.class_names)
    if "capsnet" in config["model"]:
        if "3d" in config["model"]:
            capsConfig = Config(
                cnn_in_channels=1,
                input_width=config["slice_width"],
                input_height=config["slice_width"],
                reconstruction_loss_factor=config["reconstruction_loss_factor"],
                dc_num_capsules=num_classes,
                out_capsule_size=config["out_capsule_size"],
                class_weights=ds.get_class_weights_inverse().to(device)
                if config["class_imbalance"] == "class_weights"
                else None,
                num_iterations=config["iterations"],
            )
            from models.capsule_net_3d import CapsNet

            return CapsNet(capsConfig)
        elif "2d" in config["model"]:
            capsConfig = Config(
                cnn_in_channels=1,
                input_width=config["slice_width"],
                input_height=config["slice_width"],
                reconstruction_loss_factor=config["reconstruction_loss_factor"],
                dc_num_capsules=num_classes,
                out_capsule_size=config["out_capsule_size"],
                class_weights=ds.get_class_weights_inverse().to(device)
                if config["class_imbalance"] == "class_weights"
                else None,
                num_iterations=config["iterations"],
                input_slices=1,
            )
            from models.capsule_net import CapsNet

            return CapsNet(capsConfig)
        else:
            raise Exception(f'Model {config["model"]} not found')
    elif "resnet" in config["model"]:
        if "2d" in config["model"]:
            from models.resnet import Resnet

            resnetConfig = ResnetConfig(
                num_classes,
                class_weights=ds.get_class_weights_inverse().to(device)
                if config["class_imbalance"] == "class_weights"
                else None,
            )
            return Resnet(resnetConfig)
        else:
            raise Exception(f'Model {config["model"]} not found')
    else:
        raise Exception(f'Model not found {config["model"]}')


def get_dataset(config: dict) -> BaseDataset:
    IS_3D = "3d" in config["model"]
    IS_2D = "2d" in config["model"]
    IS_CAPSNET = "capsnet" in config["model"]
    IS_RESNET = "resnet" in config["model"]
    DATASET: Literal["lungpetctx", "mnist"] = config["dataset"]

    if IS_3D:
        if DATASET == "lungpetctx":
            from preprocessing.preprocessing3D import Grayscale3D, Resize3D

            postprocess = transforms.Compose(
                [
                    Grayscale3D(),
                    Resize3D(config["slice_width"]),
                ]
            )
            return LungPetCtDxDataset_TumorClass3D(
                cache=True,
                slices_per_sample=config["slices_per_sample"],
                samples_per_scan=config["samples_per_scan"],
                postprocess=postprocess,
                exclude_classes=[
                    "Small Cell Carcinoma",
                    "Large Cell Carcinoma",
                ],
                sampling=config["class_imbalance"],
                max_size=config["max_loaded_samples"],
            )
        else:
            raise Exception(
                f'Dataset {DATASET} not available for model {config["mode"]}'
            )
    elif IS_2D:
        if DATASET == "lungpetctx":
            postprocess = transforms.Compose(
                [
                    *([transforms.Grayscale()] if IS_CAPSNET else []),
                    transforms.Resize(config["slice_width"]),
                ]
            )
            return LungPetCtDxDataset_TumorClass(
                post_normalize_transform=postprocess,
                normalize=NormalizationMethods.SINGLE_IMAGE,
                cache=True,
                exclude_classes=[
                    "Small Cell Carcinoma",
                    "Large Cell Carcinoma",
                ],
                sampling=config["class_imbalance"],
                max_size=config["max_loaded_samples"],
            )
        elif DATASET == "mnist":
            if config["slice_width"] != 28:
                print(
                    f'Warning: you are using the mnist dataset with a slice_width (here image width) of {config["slice_width"]} which is not the default of 28.'
                )
            return MNISTDataset(
                image_width=config["slice_width"],
                color_channels=3 if "resnet" in config["model"] else 1,
            )
        else:
            raise Exception(
                f'Dataset {DATASET} not available for model {config["mode"]}'
            )
    else:
        raise Exception(f'No matching dataset for model { config["model"]}')


def run_train_experiment(config: dict[str, Any], use_wandb: bool):
    ds = get_dataset(config)

    trainSet, valSet = ds.split(0.2)
    dataloaders = {
        "val": DataLoader(
            valSet, batch_size=config["batch_size"], shuffle=True, num_workers=4
        ),
        "train": DataLoader(
            trainSet, batch_size=config["batch_size"], shuffle=True, num_workers=4
        ),
    }

    torch.cuda.empty_cache()

    model = get_model(config, ds)

    wandb_watch(model)
    model.to(device)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=config["learning_rate"])

    def on_epoch_done(epoch: int, history: EpochHistory):
        wandb.log(
            {
                "epoch_train_loss": history.get_loss("train"),
                "epoch_train_accuracy": history.get_accuracy("train"),
                "epoch_val_loss": history.get_loss("val"),
                "epoch_val_accuracy": history.get_accuracy("val"),
            }
        )

    def should_stop(history: TrainHistory):
        if len(history.epoch_histories) < 3:
            return False

        combinded_loss = (
            lambda epoch: history[epoch].get_loss("val").get_combined_loss()
        )
        has_improved = combinded_loss(-3) > combinded_loss(-2) or combinded_loss(
            -3
        ) > combinded_loss(-1)
        return not has_improved

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    (model, history) = train_model(
        model,
        exp_lr_scheduler,
        dataloaders["train"],
        dataloaders["val"],
        num_epochs=config["epochs"],
        on_epoch_done=on_epoch_done,
        should_stop=should_stop if config["early_stopping"] else lambda _: False,
    )
    store_experiment(model, history, config, ds, use_wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--epochs", type=int, default=1)
    common_parser.add_argument("--max_loaded_samples", type=int, default=-1)
    common_parser.add_argument("--learning_rate", type=float, default=0.1)
    common_parser.add_argument("--slice_width", type=int, default=128)
    common_parser.add_argument("--batch_size", type=int, default=32)
    common_parser.add_argument("--iterations", type=int, default=5)
    common_parser.add_argument("--wandb", action="store_true")
    common_parser.add_argument("--early_stopping", action="store_true", default=False)
    common_parser.add_argument(
        "--class_imbalance",
        type=str,
        choices=["class_weights", "undersample", "none"],
        default="none",
    )
    common_parser.add_argument(
        "--dataset", type=str, choices=["lungpetctx", "mnist"], default="lungpetctx"
    )

    subparsers = parser.add_subparsers(required=True, dest="model")

    parser_capsnet2d_parser = subparsers.add_parser(
        "capsnet_2d", parents=[common_parser]
    )
    parser_capsnet2d_parser.add_argument("--out_capsule_size", type=int, default=16)
    parser_capsnet2d_parser.add_argument(
        "--reconstruction_loss_factor", type=float, default=0.5
    )

    parser_capsnet3d_parser = subparsers.add_parser(
        "capsnet_3d", parents=[common_parser]
    )
    parser_capsnet3d_parser.add_argument("--out_capsule_size", type=int, default=16)
    parser_capsnet3d_parser.add_argument("--slices_per_sample", type=int, default=30)
    parser_capsnet3d_parser.add_argument("--samples_per_scan", type=int, default=4)
    parser_capsnet3d_parser.add_argument(
        "--reconstruction_loss_factor", type=float, default=0.5
    )
    parser_resnet2d_parser = subparsers.add_parser("resnet_2d", parents=[common_parser])

    args = parser.parse_args()
    argsDict = vars(args)
    print(argsDict)

    use_wandb = argsDict["wandb"]
    del argsDict["wandb"]

    run = wandb.init(config=argsDict, mode="disabled" if not use_wandb else "online")
    if not run:
        raise Exception("Wandb returned a None type run.")
    with run:
        run_train_experiment(config=argsDict, use_wandb=use_wandb)
