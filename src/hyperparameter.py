import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from lungpetctdx_dataset import LungPetCtDxDataset_TumorPresence
from utils.wandb import start_wandb_run, wandb_watch, wandb_log
from eval.reconstruction_viusalization import compare_images
from utils.wandb import wandb_log, wandb
import argparse

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    
cpu = torch.device("cpu")

torch.cuda.list_gpu_processes()


IMAGE_RESOLUTION = 128
from lungpetctdx_dataset import LungPetCtDxDataset_TumorPresence
from ct_dataset import NormalizationMethods
postprocess = transforms.Compose([
    transforms.Grayscale()
])
ds = LungPetCtDxDataset_TumorPresence(post_normalize_transform=postprocess,
    normalize=NormalizationMethods.SINGLE_IMAGE, cache=True)

trainSet, valSet = ds.subject_split(0.2)#random_split(ds, [0.8, 0.2])
dataloaders = {
    "val": torch.utils.data.DataLoader(valSet, batch_size=64, shuffle=True, num_workers=4),
    "train": torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=4),
}
num_classes = len(ds.class_names)
dataset_sizes = {"train": len(trainSet),"val": len(valSet)}
# model = torch.load("test")


from capsule_net import CapsNet
from capsnet_config import Config
from train import train_model, plot_train_losses, predicted_indices_from_outputs


def run_train_experiment(config: dict = None):
    with wandb.init(config=config):
        torch.cuda.empty_cache()
        config = wandb.config
        capsConfig = Config(
            cnn_in_channels=1,
            input_width=IMAGE_RESOLUTION,
            input_height=IMAGE_RESOLUTION,
            dc_in_channels=392,
            reconstruction_loss_factor=config.reconstruction_loss_factor,
            dc_num_capsules=num_classes,
            out_capsule_size=config.out_capsule_size,
            # Num labelled 0 tensor(5682)
            # Num labelled 1 tensor(47677)
            class_weights=torch.tensor([1.0, 5682 / 47677.0]).to(device),
            num_iterations=config.iterations
        )

        model = CapsNet(capsConfig)

        wandb_watch(model)
        model.to(device)
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        (
            model,
            best_acc,
            best_loss,
            train_losses,
            best_acc_y_true,
            best_acc_y_pred,
        ) = train_model(
            model,
            exp_lr_scheduler,
            dataloaders["train"],
            dataloaders["val"],
            num_epochs=3,
            on_epoch_done=lambda epoch_result: wandb_log(epoch_result),
            on_batch_done=lambda batch_result: wandb_log(batch_result),
        )

        if best_acc_y_pred is not None and best_acc_y_true is not None:
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        preds=best_acc_y_pred.tolist(),
                        y_true=best_acc_y_true.tolist(),
                        class_names=ds.class_names,
                    )
                }
            )
        del model
        return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--out_capsule_size", type=float, required=True)
    parser.add_argument("--reconstruction_loss_factor", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()

    run_train_experiment(config=vars(args))