import torch.nn as nn
import torchvision.models as tvm

import config


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = tvm.resnet50(weights="DEFAULT" if pretrained else None)
        self.feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # [B, 2048]


class ClassificationHead(nn.Module):
    def __init__(self, in_features, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(x)


class BreastCancerModel(nn.Module):
    def __init__(self, pretrained=True, dropout=config.DROPOUT):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained)
        self.head = ClassificationHead(self.encoder.feature_dim, dropout=dropout)

    def forward(self, x, return_features=False):
        feat = self.encoder(x)
        logit = self.head(feat)
        if return_features:
            return logit, feat
        return logit

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = True