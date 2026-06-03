"""
Three model variants — controlled by config.MODE:

  "image"   → ResNet50 → SpatialAttention → GAP → FC(256) → Head
  "tabular" → TabularMLP(64) → Head
  "fused"   → image(256) ⊕ tabular(64) → Head

ImageBranch ve TabularMLP ortak; sadece üst seviye model değişir.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import config
from tabular_encoder import TabularMLP, TABULAR_EMB_DIM


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: davranış aynen korunur — eğitim/değerlendirme bozulmaz.
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.bn(self.conv(scale)))
        return x * scale

    # --- Heatmap eklentileri (eğitim/forward'u ETKİLEMEZ) -----------------
    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Sadece sigmoid attention haritasını döndürür → [B, 1, H, W]."""
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.bn(self.conv(scale)))

    def forward_with_attention(self, x: torch.Tensor):
        """attended_features, attention_map döndürür."""
        attn = self.get_attention_map(x)
        return x * attn, attn


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = tvm.resnet50(weights="DEFAULT" if pretrained else None)
        self.feature_dim = backbone.fc.in_features   # 2048
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ImageBranch(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 256,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.backbone  = ResNet50Backbone(pretrained=pretrained)
        self.attention = SpatialAttention(kernel_size=7)
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.proj      = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        feat_map = self.attention(feat_map)
        pooled   = self.pool(feat_map)
        return self.proj(pooled)

    def forward_with_attention(self, x: torch.Tensor):
        """forward(x) ile sayısal olarak aynı embedding'i + attention haritasını döndürür."""
        feat_map = self.backbone(x)
        attended, attn = self.attention.forward_with_attention(feat_map)
        pooled = self.pool(attended)
        return self.proj(pooled), attn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = config.DROPOUT,
                 hidden: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# 1) Image-only model
# ---------------------------------------------------------------------------
class ImageOnlyModel(nn.Module):
    """Sadece CNN — tabular kolu yok."""
    IMAGE_EMB_DIM = 256

    def __init__(self, pretrained: bool = True,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.image_branch = ImageBranch(
            pretrained=pretrained, out_dim=self.IMAGE_EMB_DIM, dropout=dropout,
        )
        self.head = ClassificationHead(self.IMAGE_EMB_DIM, dropout=dropout)

    def forward(self, images, cat_inputs=None, num_inputs=None):
        # cat_inputs/num_inputs imzayı diğer modellerle uyumlu tutmak için var,
        # bu modda kullanılmaz.
        img_emb = self.image_branch(images)
        return self.head(img_emb)

    def forward_with_attention(self, images, cat_inputs=None, num_inputs=None):
        """logits, attention_map döndürür (cat/num imza uyumu için, kullanılmaz)."""
        img_emb, attn = self.image_branch.forward_with_attention(images)
        return self.head(img_emb), attn

    def freeze_backbone(self):
        for p in self.image_branch.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.image_branch.backbone.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# 2) Tabular-only model
# ---------------------------------------------------------------------------
class TabularOnlyModel(nn.Module):
    """Sadece MLP — görüntü kolu yok."""

    def __init__(self, cat_dims: dict, dropout: float = config.DROPOUT):
        super().__init__()
        self.tabular_branch = TabularMLP(cat_dims=cat_dims, dropout=dropout)
        self.head = ClassificationHead(TABULAR_EMB_DIM, dropout=dropout,
                                       hidden=64)

    def forward(self, images=None, cat_inputs=None, num_inputs=None):
        # `images` ortak imza için — burada kullanılmaz.
        tab_emb = self.tabular_branch(cat_inputs, num_inputs)
        return self.head(tab_emb)

    # Backbone yok — no-op'lar diğer modellerle aynı API'yi sağlar.
    def freeze_backbone(self):
        pass

    def unfreeze_backbone(self):
        pass


# ---------------------------------------------------------------------------
# 3) Fused model (image + tabular) — mevcut davranış aynen korunur
# ---------------------------------------------------------------------------
class BreastCancerModel(nn.Module):
    IMAGE_EMB_DIM = 256

    def __init__(self, pretrained: bool = True,
                 cat_dims: dict | None = None,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.image_branch = ImageBranch(
            pretrained=pretrained, out_dim=self.IMAGE_EMB_DIM, dropout=dropout
        )
        self.use_tabular = cat_dims is not None
        if self.use_tabular:
            self.tabular_branch = TabularMLP(cat_dims=cat_dims, dropout=dropout)
            fusion_dim = self.IMAGE_EMB_DIM + TABULAR_EMB_DIM
        else:
            fusion_dim = self.IMAGE_EMB_DIM
        self.head = ClassificationHead(fusion_dim, dropout=dropout)

    def forward(self, images, cat_inputs=None, num_inputs=None):
        img_emb = self.image_branch(images)
        if self.use_tabular and cat_inputs is not None and num_inputs is not None:
            tab_emb = self.tabular_branch(cat_inputs, num_inputs)
            fused   = torch.cat([img_emb, tab_emb], dim=1)
        else:
            fused = img_emb
        return self.head(fused)

    def forward_with_attention(self, images, cat_inputs=None, num_inputs=None):
        """logits, attention_map döndürür. forward() ile aynı füzyon mantığı."""
        img_emb, attn = self.image_branch.forward_with_attention(images)
        if self.use_tabular and cat_inputs is not None and num_inputs is not None:
            tab_emb = self.tabular_branch(cat_inputs, num_inputs)
            fused   = torch.cat([img_emb, tab_emb], dim=1)
        else:
            fused = img_emb
        return self.head(fused), attn

    def freeze_backbone(self):
        for p in self.image_branch.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.image_branch.backbone.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Factory — config.MODE'a göre doğru modeli döndürür
# ---------------------------------------------------------------------------
def build_model(mode: str, cat_dims: dict | None = None,
                pretrained: bool = True,
                dropout: float = config.DROPOUT) -> nn.Module:
    """
    mode: "image" | "tabular" | "fused"
    cat_dims: tabular preprocessor'dan gelen kategorik kardinaliteler
              ("image" modunda gerekmez)
    """
    if mode == "image":
        return ImageOnlyModel(pretrained=pretrained, dropout=dropout)
    elif mode == "tabular":
        if cat_dims is None:
            raise ValueError("Tabular mode requires cat_dims.")
        return TabularOnlyModel(cat_dims=cat_dims, dropout=dropout)
    elif mode == "fused":
        if cat_dims is None:
            raise ValueError("Fused mode requires cat_dims.")
        return BreastCancerModel(pretrained=pretrained,
                                 cat_dims=cat_dims, dropout=dropout)
    else:
        raise ValueError(f"Unknown MODE: {mode!r}. "
                         f"Expected 'image', 'tabular', or 'fused'.")