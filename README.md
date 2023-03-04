# Capsulenet Tumor Classification
The main purpose of this repository is providing a pipeline for training and evaluating capsule nets on 2d or 3d image/volumetric datasets for the classification task. 
In our case we apply the pipeline on the [Lung-PET-CT-Dx dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216) as well as on [MNIST](http://yann.lecun.com/exdb/mnist/) for checking our pipeline.

## Installing requirements
### Prerequisites
- conda 
### Create conda env
Run `conda create --name <env> --file environment.yml` from this folder.

TBD
### Update conda requirements
Run `conda env export > environment.yml`

# Applications
The interface for using this repository is `src/train.py`. Always execute it from `src` (run `cd src` from the project root dir).
Via command line arguments you can specify which model to train on which dataset and other hyperparameters.
## Arguments
In the following we will describe the main arguments you can use for running the pipeline. 
In general the command structure is `python train.py [MODEL] [PARAMETER_1] ...` with `[MODEL] = resnet_2d, capsnet_2d or capsnet_3d`.
- `--dataset=[DATASET_NAME]`: Specifies the dataset. Can be `lungpetctx` or ``mnist`
- `--slice_width=[INTEGER]` Specifies the CT image width per slice OR image width in MNIST
- `--early_stopping` If set, training will stop once validation loss has not improved over 2 epochs
- `--class_imbalance=[STRATEGY]` Specifies how to deal with class imbalances (only available for lungpetctx dataset). Strategy can be "class_weights" (scale loss by inverse class frequency), "undersample" (take less samples of majority class to achieve class balance), "none" (do nothing)
Other arguments can be seen via `python train.py [MODEL] -h` (e.g. `python train.py capsnet_2d -h`)
## Examples
### Train on MNIST
To make sure our pipeline works one can train `capsnet_2d` or `resnet_2d` on the MNIST dataset.
#### Resnet
Run `python train.py resnet_2d --epochs=10 --slice_width=28 --learning_rate=0.001 --dataset=mnist`
#### CapsuleNet
Run `python train.py capsnet_2d --dataset=mnist --slice_width=128 --batch_size=4 --learning_rate=0.001`
### Train on Lung-PET-CT-Dx
#### Resnet 2d
Run `python train.py resnet_2d --epochs=10 --slice_width=28 --learning_rate=0.001 --dataset=lungpetctx --early_stopping`
#### CapsuleNet 2d
Run `python train.py capsnet_2d --epochs=10 --slice_width=28 --learning_rate=0.01 --dataset=lungpetctx`
#### Resnet 3d
Not supported
#### CapsuleNet 3d
Run `python train.py capsnet_3d --epochs=10 --slice_width=128 --learning_rate=0.01 --dataset=lungpetctx --iterations=3 --early_stopping --batch_size=4`
