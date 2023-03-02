from tqdm.notebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils.wandb import wandb_log
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Callable, Union, Tuple, Optional, Literal, TypedDict


def predicted_indices_from_outputs(outputs):
    classes = torch.sqrt((outputs**2).sum(2))
    _, max_length_indices = classes.max(dim=1)
    return torch.squeeze(max_length_indices, -1)


class LossEntry:
    def __init__(
        self,
        combined_loss: float,
        classification_loss: float,
        reconstruction_loss: float,
    ):
        self.losses = np.array(
            [combined_loss, classification_loss, reconstruction_loss]
        )

    @classmethod
    def from_losses(cls, losses: np.ndarray):
        return cls(losses[0], losses[1], losses[2])

    def get_combined_loss(self):
        return self.losses[0]

    def get_classification_loss(self):
        return self.losses[1]

    def get_reconstruction_loss(self):
        return self.losses[2]

    def __iadd__(self, other):
        assert isinstance(other, LossEntry), "Can only add another LossEntry instance"
        self.losses += other.losses
        return self

    def __add__(self, other):
        assert isinstance(other, LossEntry), "Can only add another LossEntry instance"
        return LossEntry.from_losses(self.losses + other.losses)

    def __truediv__(self, other):
        assert (
            isinstance(other, LossEntry)
            or isinstance(other, float)
            or isinstance(other, int)
        ), "Can only divide by another LossEntry instance or number"
        return LossEntry.from_losses(
            self.losses / (other.losses if isinstance(other, LossEntry) else other)
        )


class EpochPhaseHistory(TypedDict):
    losses: list[LossEntry]
    predicted_indices: np.ndarray
    label_indices: np.ndarray
    running_corrects: int
    running_total: int
    loss_sum: LossEntry


EpochPhase = Literal["train", "val"]


class EpochHistory:
    def __init__(self):
        self.history: dict[EpochPhase, EpochPhaseHistory] = {
            "train": self._initial_phase_dict(),
            "val": self._initial_phase_dict(),
        }

    def _initial_phase_dict(self) -> EpochPhaseHistory:
        return {
            "losses": [],
            "predicted_indices": np.array([]),
            "label_indices": np.array([]),
            "running_corrects": 0,
            "running_total": 0,
            "loss_sum": LossEntry(0.0, 0.0, 0.0),
        }

    def append(
        self,
        phase: EpochPhase,
        loss: LossEntry,
        batch_predictions: torch.Tensor,
        batch_labels: torch.Tensor,
    ):
        phase_history = self.history[phase]
        phase_history["losses"].append(loss)
        phase_history["predicted_indices"] = np.concatenate(
            (phase_history["predicted_indices"], batch_predictions)
        )
        phase_history["label_indices"] = np.concatenate(
            (phase_history["label_indices"], batch_labels)
        )
        batch_num_correct = torch.sum(batch_predictions == batch_labels).item()
        batch_num_total = len(batch_predictions)
        phase_history["running_corrects"] += int(batch_num_correct)
        phase_history["running_total"] += batch_num_total
        phase_history["loss_sum"] += loss

    def get_predictions_and_labels(self, phase: EpochPhase):
        """Returns predicted indiced and labels"""
        return (
            self.history[phase]["predicted_indices"],
            self.history[phase]["label_indices"],
        )

    def get_count_per_class(self, phase: EpochPhase, num_classes: int):
        labels = self.history[phase]["label_indices"]
        counts = [0] * num_classes
        for label in labels:
            assert int(label) < len(
                counts
            ), f"Label {int(label)} is not valid for num_classes {num_classes}"
            counts[int(label)] += 1
        return counts

    def get_num_corrects(self, phase: EpochPhase):
        return (
            self.history[phase]["running_corrects"],
            self.history[phase]["running_total"],
        )

    def get_accuracy(self, phase: EpochPhase):
        correct, total = self.get_num_corrects(phase)
        return correct / total

    def get_loss(self, phase: EpochPhase):
        return self.history[phase]["loss_sum"] / len(self.history[phase]["losses"])

    def __getitem__(self, phase: EpochPhase):
        return self.history[phase]


class TrainHistory:
    def __init__(self):
        self.epoch_histories: list[EpochHistory] = []

    def append(
        self,
        epoch: int,
        phase: EpochPhase,
        loss: LossEntry,
        batch_predictions_indices: torch.Tensor,
        batch_labels_indices: torch.Tensor,
    ):
        assert (
            epoch < len(self.epoch_histories) + 1
        ), f"Can't log epoch {epoch}. Only {len(self.epoch_histories)} losses tracked so far"
        if(len(batch_predictions_indices.shape) < 1):
            return
        l_len = len(batch_labels_indices)
        p_len = len(batch_predictions_indices)
        assert (
            l_len == p_len
        ), f"Length of labels ({l_len}) different to length of predictions ({p_len}) "
        if epoch == len(self.epoch_histories):
            self.epoch_histories.append(EpochHistory())
        self.epoch_histories[epoch].append(
            phase, loss, batch_predictions_indices, batch_labels_indices
        )

    def __getitem__(self, index: int):
        return self.epoch_histories[index]

    def get_best_accuracy_epoch_index(self, phase: EpochPhase):
        highest_acc = 0.0
        best_epoch = -1
        for i, epoch_history in enumerate(self.epoch_histories):
            accuracy = epoch_history.get_accuracy(phase)
            if accuracy > highest_acc:
                best_epoch = i
                highest_acc = accuracy
        return best_epoch

    def get_best_accuracy_epoch(self, phase: EpochPhase):
        return self[self.get_best_accuracy_epoch_index(phase)]


def generic_train_model(
    model: nn.Module,
    scheduler: _LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    process_batch: Callable[[int, EpochPhase, int, Tuple], None],
    on_epoch_start: Optional[Callable[[int, EpochPhase], None]] = None,
    on_epoch_done: Optional[Callable[[int, EpochPhase], None]] = None,
    num_epochs=2,
):
    since = time.time()
    dataloaders = {"train": train_loader, "val": val_loader}
    phases: list[EpochPhase] = ["train", "val"]

    for epoch in range(num_epochs):
        for phase in phases:
            if on_epoch_start:
                on_epoch_start(epoch, phase)
            if phase == "train":
                model.train()
            else:
                model.eval()

            for idx, batch in enumerate(dataloaders[phase]):
                process_batch(epoch, phase, idx, batch)

            if phase == "train":
                scheduler.step()
            if on_epoch_done:
                on_epoch_done(epoch, phase)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return model


from models.model_with_loss import ModelWithLoss


def train_model(
    model: ModelWithLoss,
    scheduler: _LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs=2,
    on_epoch_done: Optional[Callable[[int, EpochHistory], None]] = None,
):
    optimizer = scheduler.optimizer
    best_model_wts = copy.deepcopy(model.state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = {"train": train_loader, "val": val_loader}
    history = TrainHistory()

    def process_batch(epoch: int, phase: EpochPhase, idx: int, batch: Tuple):
        inputs, labels, bbox = batch
        inputs = inputs.to(device)
        reconstruction_target_images = inputs
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(phase == "train"):

            outputs, reconstructions, _ = model(inputs)
            preds = predicted_indices_from_outputs(outputs)

            loss, classification_loss, reconstruction_loss = model.loss(
                reconstruction_target_images, outputs, labels, reconstructions
            )
            # backward + optimize only if in training phase
            if phase == "train":
                loss.backward()
                optimizer.step()

        # statistics
        batch_losses = LossEntry(
            loss.item(), classification_loss.item(), reconstruction_loss.item()
        )
        _, label_indices = torch.max(labels.data, 1)

        history.append(epoch, phase, batch_losses, preds.cpu(), label_indices.cpu())

        epoch_history = history[epoch]
        accuracy = epoch_history.get_accuracy(phase)
        running_epoch_loss = epoch_history.get_loss(phase)
        tqdm.write(
            "Epoch: [{}/{}], Batch: [{}/{}], batch loss: {:.4f}| RUNNING  acc: {:.4f}, combined l.: {:.4f}, class. l.: {:.4f}, reconstr. l.: {:.4f}".format(
                epoch + 1,
                num_epochs,
                idx + 1,
                len(dataloaders[phase]),
                loss.item(),
                accuracy,
                running_epoch_loss.get_combined_loss(),
                running_epoch_loss.get_classification_loss(),
                running_epoch_loss.get_reconstruction_loss(),
            ),
            end="\r",
        )

    def _on_epoch_done(epoch: int, phase: EpochPhase):
        if phase == "val":
            best_epoch_index = history.get_best_accuracy_epoch_index(phase)
            if best_epoch_index == epoch:
                nonlocal best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

            if on_epoch_done:
                on_epoch_done(epoch, history[epoch])

    def on_epoch_start(epoch: int, phase: EpochPhase):
        print(f"\nStarting {phase} for epoch {epoch}")

    generic_train_model(
        model,
        scheduler,
        train_loader,
        val_loader,
        process_batch,
        on_epoch_start,
        _on_epoch_done,
        num_epochs=num_epochs,
    )
    model.load_state_dict(best_model_wts)
    return model, history


def plot_losses(history: TrainHistory, out_path: Optional[str] = None):
    fig, axis = plt.subplots(4, 1, figsize=(12, 12))

    def plot_any(plot_index: int, values: list[float], line_label: str):
        x = range(len(values))
        (line,) = axis[plot_index].plot(x, values)
        line.set_label(line_label)
        axis[plot_index].set_xticks(x, [str(i) for i in x])

    def plotLoss(
        plot_index: int, extract_loss: Callable[[LossEntry], float], label: str
    ):
        phases: list[EpochPhase] = ["train", "val"]
        for phase in phases:
            losses = [
                extract_loss(epoch_history.get_loss(phase))
                for epoch_history in history.epoch_histories
            ]
            plot_any(plot_index, losses, phase)
        axis[plot_index].set_title(label)

    plotLoss(0, lambda entry: entry.get_combined_loss(), "Combined Training losses")
    plotLoss(1, lambda entry: entry.get_classification_loss(), "Classification losses")
    plotLoss(2, lambda entry: entry.get_reconstruction_loss(), "Reconstruction losses")
    plot_any(
        3,
        [
            epoch_history.get_accuracy("train")
            for epoch_history in history.epoch_histories
        ],
        "train",
    )
    plot_any(
        3,
        [
            epoch_history.get_accuracy("val")
            for epoch_history in history.epoch_histories
        ],
        "val",
    )
    axis[3].set_title("Accuracy")
    axis[0].legend(ncol=1)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path)
    else:
        fig.show()
