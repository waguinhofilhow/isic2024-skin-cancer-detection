import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

class ISICModelEff(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = efficientnet_b0(
            weights=weights
        )

        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features,
            1
        )


    def forward(self, x):

        return self.backbone(x).squeeze(1)
    
class ISICModelRsn(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(
            weights=weights
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(in_features, 1)


    def forward(self, x):

        return self.backbone(x).squeeze(1)