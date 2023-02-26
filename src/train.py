import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from dataset.lungpetctdx_dataset import LungPetCtDxDataset_TumorPresence
from utils.wandb import wandb_watch, wandb_log
from utils.wandb import wandb_log, wandb
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    
cpu = torch.device("cpu")

torch.cuda.list_gpu_processes()


IMAGE_RESOLUTION = 128
from dataset.lungpetctdx_dataset import LungPetCtDxDataset_TumorPresence
from dataset.ct_dataset import NormalizationMethods

# model = torch.load("test")
from models.capsule_net import CapsNet
from model_config.capsnet_config import Config
from utils.train import train_model, EpochHistory, plot_losses
from wandb.plot import confusion_matrix
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

def run_train_experiment(config: dict, use_wandb: bool):
    postprocess = transforms.Compose([
        transforms.Grayscale()
    ])
    ds = LungPetCtDxDataset_TumorPresence(post_normalize_transform=postprocess,
        normalize=NormalizationMethods.SINGLE_IMAGE, cache=True)

    trainSet, valSet = ds.subject_split(0.2)
    dataloaders = {
        "val": DataLoader(valSet, batch_size=config["batch_size"], shuffle=True, num_workers=4),
        "train": DataLoader(trainSet, batch_size=config["batch_size"], shuffle=True, num_workers=4),
    }
    num_classes = len(ds.class_names)
    dataset_sizes = {"train": len(trainSet),"val": len(valSet)}



    torch.cuda.empty_cache()
    capsConfig = Config(
        cnn_in_channels=1,
        input_width=IMAGE_RESOLUTION,
        input_height=IMAGE_RESOLUTION,
        dc_in_channels=392,
        reconstruction_loss_factor=config["reconstruction_loss_factor"],
        dc_num_capsules=num_classes,
        out_capsule_size=config["out_capsule_size"],
        # Num labelled 0 tensor(5682)
        # Num labelled 1 tensor(47677)
        class_weights=torch.tensor([1.0, 5682 / 47677.0]).to(device),
        num_iterations=config["iterations"]
    )
    model = CapsNet(capsConfig)
    wandb_watch(model)
    model.to(device)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=config["learning_rate"])

    def on_epoch_done(epoch:int, history: EpochHistory):
        wandb.log({
            "epoch_train_loss": history.get_loss("train"), 
            "epoch_train_accuracy": history.get_accuracy("train"), 
            "epoch_val_loss": history.get_loss("val"), 
            "epoch_val_accuracy": history.get_accuracy("val"), 
        })
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    (
        model,
        history
    ) = train_model(
        model,
        exp_lr_scheduler,
        dataloaders["train"],
        dataloaders["val"],
        num_epochs=3,
        on_epoch_done=on_epoch_done
    )
    best_acc_epoch = history.get_best_accuracy_epoch("val")
    best_acc_y_pred, best_acc_y_true = best_acc_epoch.get_predictions_and_labels("val")
    if best_acc_y_pred is not None and best_acc_y_true is not None:
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
        else: 
            pass
    
    if not use_wandb:
        now = datetime.now()
        now_formatted = now.strftime("%m.%d.%Y_%H:%M:%S")
        out_dir = f"results/{now_formatted}"
        os.makedirs(out_dir, exist_okay=True)
        plot_losses(history, out_path = os.path.join(out_dir, "history.png"))

        plt.clf()
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(best_acc_y_true.tolist(), best_acc_y_pred.tolist()), display_labels=ds.class_names)
        disp.plot()
        plt.savefig(os.path.join(out_dir, "best_val_acc_conf_matrix.png"))
        with open(os.path.join(out_dir, "parameters.json"), "w") as f:
            json.dump(config.__dict__, f)
        print(f"Saved results at {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_capsule_size", type=float, required=True)
    parser.add_argument("--reconstruction_loss_factor", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dataset", type=str, choices=["lungpet_2d", "lungpet_3d"],required=True)

    subparsers = parser.add_subparsers()
    parser_lungpet_2d_parser = subparsers.add_parser("--parser_lungpet_2d")
    parser_lungpet_3d_parser = subparsers.add_parser("--parser_lungpet_3d")
    args = parser.parse_args()
    argsDict= vars(args)

    use_wandb = argsDict["wandb"]
    del argsDict["wandb"]

    if use_wandb:
        with wandb.init(config=argsDict):
            run_train_experiment(config=wandb.config, True)

    else:
        run_train_experiment(config=argsDict, False)