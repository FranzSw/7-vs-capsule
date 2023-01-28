from tqdm.notebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils.wandb import wandb_log
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from capsule_net import CapsNet
from typing import Callable, Union

def predicted_indices_from_outputs(outputs):
    classes = torch.sqrt((outputs ** 2).sum(2))
    _, max_length_indices = classes.max(dim=1)
    return torch.squeeze(max_length_indices, -1)


def train_model(model: CapsNet, scheduler: _LRScheduler, trainLoader: DataLoader,valLoader: DataLoader, num_epochs=2, on_epoch_done: Union[Callable[[dict[str, float]], None], None] = None, on_batch_done: Callable[[dict[str, float]], None] = None):
    since = time.time()
    optimizer = scheduler.optimizer
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 9999999999.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = {
        "train": trainLoader,
        "val": valLoader
    }
    dataset_sizes = {
        "train": len(trainLoader)* (trainLoader.batch_size if trainLoader.batch_size else 1),
        "val": len(valLoader) * (valLoader.batch_size if valLoader.batch_size else 1)
    }
    train_losses: list[list[tuple[float, float, float]]] = []

    best_acc_y_true = None
    best_acc_y_pred = None
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        epoch_result = {}
        epoch_train_losses: list[tuple[float, float, float]] = []

        epoch_y_true = np.array([])
        epoch_y_pred = np.array([])
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_corrects = 0
            running_losses = np.array([0,0,0], np.float32)

            # Iterate over data.
            for idx, (inputs, labels, bounding_boxes) in enumerate(dataloaders[phase]):
               
                #reconstruction_target_images = torch.tensor(list(map(mask_image, inputs, bounding_boxes)))
                #reconstruction_target_images = reconstruction_target_images.to(device)
                
                inputs = inputs.to(device)
                reconstruction_target_images =  inputs
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, reconstructions, _ = model(inputs)
                    preds = predicted_indices_from_outputs(outputs)
                    loss, classification_loss, reconstruction_loss = model.loss(reconstruction_target_images, outputs, labels, reconstructions, CEL_for_classifier=True)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            

                # statistics
                batch_size = inputs.size(0)
                batch_loss = loss.item() * batch_size
                batch_classification_loss = classification_loss.item() * batch_size
                batch_reconstruction_loss = reconstruction_loss.item() * batch_size
                batch_losses = (batch_loss,batch_classification_loss, batch_reconstruction_loss)
                if phase=="train":
                    epoch_train_losses.append(batch_losses)
                

                running_losses += np.array(batch_losses)
                _, labels_index = torch.max(labels.data, 1)
                batch_num_correct = torch.sum(preds ==labels_index)
                batch_accuracy = batch_num_correct / float(batch_size)
                running_corrects += batch_num_correct
                if phase == "val":
                    y_true = np.concatenate((epoch_y_true, labels_index.cpu()))
                    y_pred = np.concatenate((epoch_y_pred, preds.cpu()))

                if idx % 10 == 0:
                    batch_size = inputs.size(0)
                    tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train acc: {:.6f}, batch loss: {:.6f}, running reconstr. loss: {:.6f}, running class. loss: {:.6f}".format(
                        epoch + 1,
                        num_epochs,
                        idx + 1,
                        len(dataloaders[phase]),
                        batch_accuracy,
                        batch_loss, 
                        running_losses[2]/(idx+1), 
                        running_losses[1]/(idx+1)
                        ), end="\r")

                if on_batch_done:
                    batch_result = {}
                    batch_result[phase] = {"batch_accuracy": batch_accuracy, "batch_loss": batch_loss, "batch_reconstruction_loss": batch_reconstruction_loss, "batch_classification_loss": batch_classification_loss}
                    on_batch_done(batch_result)
            if phase == 'train':
                scheduler.step()
                train_losses.append(epoch_train_losses)

            epoch_loss: float = running_losses[0] / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            epoch_reconstruction_loss = running_losses[2] / dataset_sizes[phase]
            epoch_classification_loss =  running_losses[1]/ dataset_sizes[phase]

            epoch_result[phase] = {"epoch_acc": epoch_acc, "epoch_loss": epoch_loss, "epoch_classification_loss": epoch_classification_loss, "epoch_reconstruction_loss":epoch_reconstruction_loss}

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_acc_y_true = epoch_y_true
                    best_acc_y_pred = epoch_y_pred
                if epoch_loss < best_loss:
                    best_loss = epoch_loss

        if on_epoch_done:
            on_epoch_done(epoch_result)
        print()


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_loss,train_losses, best_acc_y_true, best_acc_y_pred

def plot_train_losses(losses: list[list[tuple[float, float, float]]]):
    figure, axis = plt.subplots(3)
    def plotLoss(lossIndex: int, title: str, epoch:int):
        line, = axis[lossIndex].plot(range(len(losses)), [entry[lossIndex] for entry in losses[epoch]])
        line.set_label(f"Epoch {epoch}")
        axis[lossIndex].set_title(title)
    
    for epoch in range(len(losses)):
        for i, title in enumerate(["Combined Training losses", "Classification losses", "Reconstruction losses"]):
            plotLoss(i, title, epoch)