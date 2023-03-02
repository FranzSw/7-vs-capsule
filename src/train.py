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
from sklearn.metrics import confusion_matrix as conf_matrix_sklearn, ConfusionMatrixDisplay, f1_score

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    
cpu = torch.device("cpu")

torch.cuda.list_gpu_processes()
from dataset.lungpetctdx_dataset import LungPetCtDxDataset_TumorClass, LungPetCtDxDataset_TumorClass3D
from dataset.ct_dataset import NormalizationMethods

# model = torch.load("test")
from model_config.capsnet_config import Config
from model_config.ResnetConfig import ResnetConfig
from utils.train import train_model, EpochHistory, plot_losses
from wandb.plot import confusion_matrix
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

def run_train_experiment(config: dict, use_wandb: bool):

    IS_3D = '3d' in config['model']
    IS_2D = '2d' in config['model']
    IS_CAPSNET = 'capsnet' in config['model']
    IS_RESNET = 'resnet' in config['model']

    if IS_3D:
        from preprocessing.preprocessing3D import Grayscale3D, Resize3D

        postprocess = transforms.Compose(
            [
                Grayscale3D(),
                Resize3D(config['slice_width']),
            ]
        )
        ds = LungPetCtDxDataset_TumorClass3D(
            cache=True,
            slices_per_sample=config['slices_per_sample'],
            samples_per_scan=config['samples_per_scan'],
            postprocess=postprocess,
            exclude_classes=[
                "Small Cell Carcinoma",
                "Large Cell Carcinoma",
            ],
            sampling=config['class_imbalance'],
            max_size=config['max_loaded_samples'],
        )
    elif IS_2D:
        postprocess = transforms.Compose([
            *([transforms.Grayscale()] if IS_CAPSNET else []),
            transforms.Resize(config['slice_width']),
        ])
        ds = LungPetCtDxDataset_TumorClass(
            post_normalize_transform=postprocess,
            normalize=NormalizationMethods.SINGLE_IMAGE,
            cache=True,
            exclude_classes=[
                "Small Cell Carcinoma",
                "Large Cell Carcinoma",
            ],
            sampling=config['class_imbalance'],
            max_size=config['max_loaded_samples'],
        )
    else:
        print('No dataset for model', config['model'])
        return

    trainSet, valSet = ds.subject_split(0.2)
    dataloaders = {
        "val": DataLoader(valSet, batch_size=config["batch_size"], shuffle=True, num_workers=4),
        "train": DataLoader(trainSet, batch_size=config["batch_size"], shuffle=True, num_workers=4),
    }
    num_classes = len(ds.class_names)

    torch.cuda.empty_cache()
    
    if 'capsnet' in config['model']:
        if '3d' in config['model']:
            capsConfig = Config(
                cnn_in_channels=1,
                input_width=config['slice_width'],
                input_height=config['slice_width'],
                reconstruction_loss_factor=config["reconstruction_loss_factor"],
                dc_num_capsules=num_classes,
                out_capsule_size=config["out_capsule_size"],
                # Num labelled 0 tensor(5682)
                # Num labelled 1 tensor(47677)
                # TODO: Calculate class weights if needed
                class_weights=ds.get_class_weights_inverse().to(device) if config['class_imbalance']=='class_weights' else None,
                num_iterations=config["iterations"]
            )
            from models.capsule_net_3d import CapsNet
        else:
            capsConfig = Config(
                cnn_in_channels=1,
                input_width=config['slice_width'],
                input_height=config['slice_width'],
                reconstruction_loss_factor=config["reconstruction_loss_factor"],
                dc_num_capsules=num_classes,
                out_capsule_size=config["out_capsule_size"],
                # Num labelled 0 tensor(5682)
                # Num labelled 1 tensor(47677)
                # TODO: Calculate class weights if needed
                class_weights=ds.get_class_weights_inverse().to(device) if config['class_imbalance']=='class_weights' else None,
                num_iterations=config["iterations"],
                input_slices=1,
            )
            from models.capsule_net import CapsNet
        model = CapsNet(capsConfig)
    elif 'resnet' in config['model']:
        from models.resnet import Resnet
        resnetConfig = ResnetConfig(
            num_classes,
            class_weights=ds.get_class_weights_inverse().to(device) if config['class_imbalance']=='class_weights' else None,
        )
        model = Resnet(resnetConfig)
    else:
        print('Model not found', config['model'])
    
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
        num_epochs=config['epochs'],
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
        os.makedirs(out_dir, exist_ok=True)
        plot_losses(history, out_path = os.path.join(out_dir, "history.png"))

        plt.clf()
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_sklearn(best_acc_y_true, best_acc_y_pred, labels=range(len(ds.class_names))), display_labels=ds.class_names)
        disp.plot()
        plt.savefig(os.path.join(out_dir, "best_val_acc_conf_matrix.png"))
        with open(os.path.join(out_dir, "parameters.json"), "w") as f:
            json.dump(config, f, indent=4)
        print(f"Saved results at {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--epochs", type=int, default=1)
    common_parser.add_argument("--max_loaded_samples", type=int, default=-1)
    common_parser.add_argument("--learning_rate", type=float, default=0.1)
    common_parser.add_argument("--slice_width", type=float, default=128)
    common_parser.add_argument("--batch_size", type=int, default=32)
    common_parser.add_argument("--iterations", type=int, default=5)
    common_parser.add_argument("--wandb", action="store_true")
    common_parser.add_argument('--class_imbalance', type=str, choices=['class_weights', 'undersample', 'none'], default='none')



    subparsers = parser.add_subparsers(required=True, dest='model')

    parser_lungpet_2d_parser = subparsers.add_parser("capsnet_2d", parents=[common_parser])
    parser_lungpet_2d_parser.add_argument("--out_capsule_size", type=float, default=16)
    parser_lungpet_2d_parser.add_argument("--reconstruction_loss_factor", type=float, default=0.5)

    parser_lungpet_3d_parser = subparsers.add_parser("capsnet_3d", parents=[common_parser])
    parser_lungpet_3d_parser.add_argument("--out_capsule_size", type=float, default=16)
    parser_lungpet_3d_parser.add_argument('--slices_per_sample', type=int, default=30)
    parser_lungpet_3d_parser.add_argument('--samples_per_scan', type=int, default=4)

    parser_lungpet_resnet_parser = subparsers.add_parser("resnet_2d", parents=[common_parser])

    # parser.add_argument("--dataset", type=str, choices=["lungpet_2d", "lungpet_3d"],required=True)

    args = parser.parse_args()
    argsDict= vars(args)
    print(argsDict)

    use_wandb = argsDict["wandb"]
    del argsDict["wandb"]

    if use_wandb:
        with wandb.init(config=argsDict):
            run_train_experiment(config=wandb.config, use_wandb=True)

    else:
        with wandb.init(config=argsDict, mode="disabled"):
            run_train_experiment(config=argsDict, use_wandb=False)