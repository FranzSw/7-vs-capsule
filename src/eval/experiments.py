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
from typing import Literal, Union
from dataset.mnist_dataset import MNISTDataset
from eval.shap import ShapEvaluation
from models.capsule_net import CapsNet
from models.capsule_net_3d import CapsNet as CapsNet3d
from models.resnet import Resnet
from models.model_with_loss import ModelWithLoss


def store_experiment(
    model: ModelWithLoss,
    history: TrainHistory,
    config: dict,
    ds: BaseDataset,
    use_wandb: bool,
):
    best_acc_epoch = history.get_best_accuracy_epoch("val")
    best_acc_y_pred, best_acc_y_true = best_acc_epoch.get_predictions_and_labels("val")

    now = datetime.now()
    now_formatted = now.strftime("%m.%d.%Y_%H:%M:%S")
    out_dir = f"results/{now_formatted}"
    os.makedirs(out_dir, exist_ok=True)
    plot_losses(history, out_path=os.path.join(out_dir, "history.png"))
    plt.clf()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix_sklearn(
            best_acc_y_true, best_acc_y_pred, labels=range(len(ds.class_names))
        ),
        display_labels=ds.class_names,
    )
    disp.plot()
    plt.savefig(os.path.join(out_dir, "best_val_acc_conf_matrix.png"))
    with open(os.path.join(out_dir, "parameters.json"), "w") as f:
        json.dump(config, f, indent=4)
    torch.save(model, os.path.join(out_dir, "model.pt"))

    print(f"Saved results at {out_dir}")

    if "2d" in config["model"]:
        print("Running SHAP evaluation")
        width = config["slice_width"]
        shap_eval = ShapEvaluation(
            model, (width, width, model.input_color_channels), ds.class_names
        )
        _, valSet = ds.split(0.2)
        loader = DataLoader(valSet, batch_size=10, shuffle=True, num_workers=4)
        inputs, *rest = next(iter(loader))
        bounding_boxes = rest[1] if len(rest) > 1 else None
        out_path = os.path.join(out_dir, "shap.png")
        shap_eval.evaluate(inputs, bounding_boxes, out_path)

        if use_wandb:
            artifact = wandb.Artifact("shap_evaluation", type="image")
            artifact.add_file(out_path)
            wandb.log_artifact(artifact)

    if use_wandb:
        wandb.log(
            {
                "confusion_matrix": confusion_matrix(
                    preds=best_acc_y_pred.tolist(),
                    y_true=best_acc_y_true.tolist(),
                    class_names=ds.class_names,
                ),
                "best_accuracy": best_acc_epoch.get_accuracy("val"),
            }
        )
