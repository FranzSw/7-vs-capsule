from dataclasses import asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split, Subset
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import copy
from lungpetctdx_dataset import LungPetCtDxDataset_TumorClass as DS

from tqdm import tqdm


import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from model_config.ResnetConfig import ResnetConfig
from utils.mask import mask_image
from utils.wandb import start_wandb_run, wandb_log, wandb_watch

def plot_confusion_matrix(dataloaders, class_names, model, suf=''):
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.cpu(), 1)
            _, labels = torch.max(classes.cpu(), 1)
            y_true = np.concatenate((y_true, preds))
            y_pred = np.concatenate((y_pred, labels))
            
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=class_names)
    disp.plot()
    plt.show()
    plt.savefig(f'conf{suf}.png')
    print("F1-score:",f1_score(y_true, y_pred, average="weighted") )



cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
if torch.cuda.is_available():
    print('Using CUDA')
else:
    print('NO CUDA ENABLED!')

def create_splitted_dataset():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
    ])

    trainSet, valSet = DS(post_normalize_transform=preprocess, exclude_classes=[DS.all_tumor_class_names[1],DS.all_tumor_class_names[2]]).subject_split(test_size=0.2)
    return trainSet, valSet

def create_dataloaders(trainSet: Subset, valSet: Subset):
    return {
        "train": torch.utils.data.DataLoader(trainSet, batch_size=128, shuffle=True, num_workers=8),
        "val": torch.utils.data.DataLoader(valSet, batch_size=128, shuffle=True, num_workers=8)
    }

def create_model(config: ResnetConfig):
    resnet = torch.hub.load(config.source, config.model_name)
    # freeze all layers but last
    for param in resnet.parameters():
        param.requires_grad = True

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, config.num_outputs)
    return resnet

def train_model(dataloaders, class_names, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Used to log results in one dict to Wandb
        result_dict = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            data_length = len(dataloaders[phase].dataset)
            number_of_batches = len(dataloaders[phase])

            # Iterate over data.
            for idx,(inputs, labels, bounding_boxes) in tqdm(enumerate(dataloaders[phase], 0), unit="batch", total=number_of_batches):
                inputs = torch.tensor(list(map(mask_image, inputs, bounding_boxes)))
 
                inputs = inputs.to(device)
                labels = labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                try:
                    running_loss += loss.item() * inputs.size(0)
                    _, labels_index = torch.max(labels.data, 1)
                    running_corrects += torch.sum(preds ==labels_index)
                    
                except RuntimeError as e:
                    print("RuntimeError was thrown while calculating statistics: ", e)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_length
            epoch_acc = running_corrects.double() / data_length

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            result_dict[phase] = {
                'loss': epoch_loss,
                'acc': epoch_acc,
            }

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                plot_confusion_matrix(dataloaders, class_names, model, epoch)

        print()
        wandb_log(result_dict)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    trainSet, valSet = create_splitted_dataset()
    dataloaders = create_dataloaders(trainSet, valSet)
    class_names = [DS.all_tumor_class_names[0], DS.all_tumor_class_names[3]]
    num_classes = len(class_names)
    # dataset_sizes = {"train": len(trainSet),"val": len(valSet)}

    resnet_config = ResnetConfig(num_outputs= len(class_names))
    resnet_config.weighted_loss = False

    resnet = create_model(resnet_config)
    resnet.to(device)

    start_wandb_run(resnet_config)
    wandb_watch(resnet)

    if resnet_config.weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=trainSet.dataset.get_class_weights_inverse().to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(resnet.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    resnet = train_model(dataloaders, class_names, resnet, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

    plot_confusion_matrix(dataloaders, class_names, resnet)



if __name__ == '__main__':
    main()