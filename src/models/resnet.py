import torch
from model_config.ResnetConfig import ResnetConfig
from models.model_with_loss import ModelWithLoss
from typing import Optional


def create_resnet_model(config):
    resnet = torch.hub.load(config.source, config.model_name)
    # unfreeze
    for param in resnet.parameters():
        param.requires_grad = True

    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, config.num_outputs), torch.nn.Softmax(-1)
    )

    return resnet


class Resnet(ModelWithLoss):
    def __init__(self, config):
        super(Resnet, self).__init__()

        self.model = create_resnet_model(config)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=config.class_weights)

    def forward(self, data):
        output = self.model(data)
        return torch.unsqueeze(output, -1), data, data

    def loss(
        self,
        input_images: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        reconstructions: Optional[torch.Tensor],
        CEL_for_classifier=False,
    ):
        classification_loss = self.ce_loss(predictions.squeeze(-1), labels)
        
        return (
            classification_loss,
            classification_loss,
            torch.tensor(0),
        )
