"""
SSD (Single Shot Multibox Detector) implementation for polyp detection.
Supports VGG16 and MobileNetV2 backbones with 300x300 and 512x512 inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, mobilenet_v2, VGG16_Weights, MobileNet_V2_Weights
from typing import List, Tuple


class SSD300_VGG16(nn.Module):
    """SSD300 with VGG16 backbone"""
    
    def __init__(self, num_classes: int = 2):  # background + polyp
        super().__init__()
        self.num_classes = num_classes
        self.image_size = 300
        
        # Load pretrained VGG16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        
        # Modify VGG for SSD (remove last pooling, add extra conv layers)
        self.features = self._modify_vgg(self.features)
        
        # Extra convolutional layers for multiscale feature maps
        self.extra_layers = self._add_extra_layers()
        
        # Prediction heads
        self.loc_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        
        # Feature map sizes and channels
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.feature_channels = [512, 1024, 512, 256, 256, 256]
        
        self._build_heads()
        
        # Initialize weights
        self._init_weights()
    
    def _modify_vgg(self, features):
        """Remove last pooling and add conv6, conv7"""
        features = features[:-1]  # remove last maxpool
        # Add conv6 (dilated) and conv7
        features.append(nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(1024, 1024, kernel_size=1))
        features.append(nn.ReLU(inplace=True))
        return features
    
    def _add_extra_layers(self):
        """Add extra convolutional layers for SSD"""
        layers = []
        # conv8_1, conv8_2
        layers.append(nn.Conv2d(1024, 256, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # conv9_1, conv9_2
        layers.append(nn.Conv2d(512, 128, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # conv10_1, conv10_2
        layers.append(nn.Conv2d(256, 128, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))
        # conv11_1, conv11_2
        layers.append(nn.Conv2d(256, 128, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))
        return nn.ModuleList(layers)
    
    def _build_heads(self):
        """Build localization and classification heads"""
        for fm_size, channels in zip(self.feature_maps, self.feature_channels):
            num_anchors = self._num_anchors_per_location(fm_size)
            loc_head = nn.Conv2d(channels, num_anchors * 4, kernel_size=3, padding=1)
            cls_head = nn.Conv2d(channels, num_anchors * self.num_classes, kernel_size=3, padding=1)
            self.loc_heads.append(loc_head)
            self.cls_heads.append(cls_head)
    
    def _num_anchors_per_location(self, feature_map_size: int) -> int:
        """Number of anchors per location for given feature map"""
        if feature_map_size in [38, 19]:  # conv4_3, conv7
            return 4
        elif feature_map_size in [10, 5, 3, 1]:  # extra layers
            return 6
        return 4
    
    def _init_weights(self):
        """Initialize head weights with xavier"""
        for head in self.loc_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.cls_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (loc_preds, cls_preds)"""
        sources = []
        
        # VGG features up to conv4_3
        for i in range(23):  # up to conv4_3
            x = self.features[i](x)
        sources.append(x)  # 38x38
        
        # VGG features up to conv7
        for i in range(23, len(self.features)):
            x = self.features[i](x)
        sources.append(x)  # 19x19
        
        # Extra layers
        for i, layer in enumerate(self.extra_layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and i % 2 == 1:  # after ReLU, conv2d odd index
                sources.append(x)
        
        loc_preds = []
        cls_preds = []
        
        for i, (src, loc_head, cls_head) in enumerate(zip(sources, self.loc_heads, self.cls_heads)):
            loc = loc_head(src).permute(0, 2, 3, 1).contiguous()
            cls = cls_head(src).permute(0, 2, 3, 1).contiguous()
            batch_size = loc.size(0)
            loc = loc.view(batch_size, -1, 4)
            cls = cls.view(batch_size, -1, self.num_classes)
            loc_preds.append(loc)
            cls_preds.append(cls)
        
        loc_preds = torch.cat(loc_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        
        return loc_preds, cls_preds


class SSD300_MobileNetV2(SSD300_VGG16):
    """SSD300 with MobileNetV2 backbone (lighter, faster)"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__(num_classes)
        self.image_size = 300
        
        # Load pretrained MobileNetV2
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = self._modify_mobilenet(mobilenet.features)
        
        # Rebuild extra layers and heads for MobileNet
        self.extra_layers = self._add_extra_layers_mobilenet()
        self.feature_channels = [320, 1280, 512, 256, 256, 128]
        self.loc_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        self._build_heads()
        self._init_weights()
    
    def _modify_mobilenet(self, features):
        """Extract MobileNet features up to appropriate layers"""
        # Keep all features, we'll take specific layers
        return features
    
    def _add_extra_layers_mobilenet(self):
        """Extra layers optimized for MobileNet"""
        layers = []
        layers.append(nn.Conv2d(1280, 256, kernel_size=1))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(512, 128, kernel_size=1))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(256, 128, kernel_size=1))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(256, 64, kernel_size=1))
        layers.append(nn.ReLU6(inplace=True))
        layers.append(nn.Conv2d(64, 128, kernel_size=3))
        layers.append(nn.ReLU6(inplace=True))
        return nn.ModuleList(layers)