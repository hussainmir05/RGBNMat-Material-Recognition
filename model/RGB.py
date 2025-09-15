import torch
import torch.nn as nn
import timm
import sys

class SwinBackboneWrapper(nn.Module):
    def __init__(self, in_channels, freeze_stage_level=0):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True
        )
        old_proj = self.backbone.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None
        )
        
        with torch.no_grad():
            new_proj.weight[:, 0:3] = old_proj.weight  # RGB
        
        self.backbone.patch_embed.proj = new_proj
        self.backbone.head = nn.Identity()
        self.feature_dim = self.backbone.num_features
        
        # Freeze stages up to freeze_stage_level
        for name, param in self.backbone.named_parameters():
            for i in range(freeze_stage_level):
                if f"layers.{i}" in name:
                    param.requires_grad = False
                    break

    def forward(self, x):
        return self.backbone.forward_features(x)  # (B, H, W, C)


class RGBStreamSwin(nn.Module):
    def __init__(self, num_classes, freeze_rgbn=2):
        super().__init__()
        self.rgbn_backbone = SwinBackboneWrapper(in_channels=3, freeze_stage_level=freeze_rgbn)

        self.rgbnclassifier = nn.Sequential(
            nn.Linear(self.rgbn_backbone.feature_dim , 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(32, num_classes)
        )

    def forward(self, rgb):

        feat_rgbn = self.rgbn_backbone(rgb)  # (B, H, W, C)

        # Global average pooling over H and W dims
        feat_rgbn = feat_rgbn.mean(dim=(1, 2))  # (B, C)
        # sys.exit()

        # rgbn and slope classifiers
        x = self.rgbnclassifier(feat_rgbn)  # (B, C)
        # print(feat_rgbn.shape) 
        # print(x.shape)
        # sys.exit()


        return self.classifier(x)


if __name__ == "__main__":
    model = RGBStreamSwin(num_classes=6, freeze_rgbn=2)

    rgbn = torch.randn(8, 3, 224, 224)   # RGB+NIR input

    out = model(rgbn)
    print("Output shape:", out.shape)  # Expected: [8, 4]

