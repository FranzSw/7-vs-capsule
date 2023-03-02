


import torch
from model_config.ResnetConfig import ResnetConfig
from models.model_with_loss import ModelWithLoss

def create_resnet_model(config):
    resnet = torch.hub.load(config.source, config.model_name)
    # unfreeze
    for param in resnet.parameters():
        param.requires_grad = True

    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, config.num_outputs)
    return resnet


class Resnet(ModelWithLoss):
    def __init__(self, config):
        super(Resnet, self).__init__()

        self.model = create_resnet_model(config)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=config.class_weights)

    def forward(self, data):
        output = self.model(data)
        return torch.unsqueeze(output, -1), data, data

    def loss(self, data, x, target, reconstructions, CEL_for_classifier=False):
        classification_loss = self.cross_entropy_loss(x, target)

        return (
            classification_loss,
            classification_loss,
            torch.tensor(0),
        )

    def cross_entropy_loss(self, x, labels):
        # output capsule vector geometrical lengths
        batch_size = x.size(0)
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True)).view(batch_size, -1)

        return self.ce_loss(torch.nn.Softmax(-1)(v_c), labels)