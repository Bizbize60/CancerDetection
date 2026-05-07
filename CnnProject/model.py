"""
Fused model:  ResNet50 → SpatialAttention → GAP → FC(256)
                                                    ↓ concat
              TabularMLP ─────────────────────────→ (256+64)
                                                    ↓
                              ClassificationHead → 1 logit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import config
from tabular_encoder import TabularMLP, TABULAR_EMB_DIM


# ---------------------------------------------------------------------------
# Spatial Attention
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """
    Channel-wise max & avg pooling → 7x7 conv → sigmoid gate.
    Applied on the last feature map of ResNet50 (B, 2048, H, W).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_out = x.mean(dim=1, keepdim=True)           # (B,1,H,W)
        max_out, _ = x.max(dim=1, keepdim=True)         # (B,1,H,W)
        scale = torch.cat([avg_out, max_out], dim=1)    # (B,2,H,W)
        scale = torch.sigmoid(self.bn(self.conv(scale)))# (B,1,H,W)
        return x * scale                                 # attended feature map


# ---------------------------------------------------------------------------
# ResNet50 backbone  (returns feature map, not pooled vector)
# ---------------------------------------------------------------------------
class ResNet50Backbone(nn.Module):
    """Return last feature map (B, 2048, H, W)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = tvm.resnet50(weights="DEFAULT" if pretrained else None)
        # Remove avgpool + fc; keep everything up to layer4
        self.feature_dim = backbone.fc.in_features   # 2048
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # (B, 2048, H, W)


# ---------------------------------------------------------------------------
# Image branch  (backbone + attention + pooling + projection)
# ---------------------------------------------------------------------------
class ImageBranch(nn.Module):
    """
    ResNet50 → SpatialAttention → AdaptiveAvgPool → Flatten → FC(256) → ReLU
    """

    def __init__(self, pretrained: bool = True, out_dim: int = 256,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.attention = SpatialAttention(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)            # (B, 2048, H, W)
        feat_map = self.attention(feat_map)    # (B, 2048, H, W) — attended
        pooled   = self.pool(feat_map)         # (B, 2048, 1, 1)
        return self.proj(pooled)               # (B, 256)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = config.DROPOUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Fused model  (image branch + tabular branch → intermediate fusion)
# ---------------------------------------------------------------------------
class BreastCancerModel(nn.Module):
    """
    Intermediate fusion:
        image_emb   (B, 256)  +  tabular_emb  (B, 64)  →  concat (B, 320)
        → ClassificationHead → (B, 1)
    """

    IMAGE_EMB_DIM = 256

    def __init__(self, pretrained: bool = True,
                 cat_dims: dict | None = None,
                 dropout: float = config.DROPOUT):
        super().__init__()
        # Image branch
        self.image_branch = ImageBranch(
            pretrained=pretrained, out_dim=self.IMAGE_EMB_DIM, dropout=dropout
        )

        # Tabular branch  (lazy: if cat_dims is None → image-only fallback)
        self.use_tabular = cat_dims is not None
        if self.use_tabular:
            self.tabular_branch = TabularMLP(cat_dims=cat_dims, dropout=dropout)
            fusion_dim = self.IMAGE_EMB_DIM + TABULAR_EMB_DIM
        else:
            fusion_dim = self.IMAGE_EMB_DIM

        # Fusion head
        self.head = ClassificationHead(fusion_dim, dropout=dropout)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        images: torch.Tensor,
        cat_inputs: dict[str, torch.Tensor] | None = None,
        num_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img_emb = self.image_branch(images)          # (B, 256)

        if self.use_tabular and cat_inputs is not None and num_inputs is not None:
            tab_emb = self.tabular_branch(cat_inputs, num_inputs)  # (B, 64)
            fused = torch.cat([img_emb, tab_emb], dim=1)           # (B, 320)
        else:
            fused = img_emb                                         # (B, 256)

        return self.head(fused)                      # (B, 1)

    # ------------------------------------------------ return intermediate feats
    def forward_with_features(
        self,
        images: torch.Tensor,
        cat_inputs: dict[str, torch.Tensor] | None = None,
        num_inputs: torch.Tensor | None = None,
    ):
        img_emb = self.image_branch(images)
        if self.use_tabular and cat_inputs is not None and num_inputs is not None:
            tab_emb = self.tabular_branch(cat_inputs, num_inputs)
            fused = torch.cat([img_emb, tab_emb], dim=1)
        else:
            fused = img_emb
        logit = self.head(fused)
        return logit, img_emb

    # ------------------------------------------------- freeze / unfreeze backbone
    def freeze_backbone(self):
        for p in self.image_branch.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.image_branch.backbone.parameters():
            p.requires_grad = True
