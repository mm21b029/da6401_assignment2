import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CustomCNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        base_filters=32,
        filter_organization="double",
        activation="relu",
        batch_norm=False,
        dropout_rate=0.2,
        dropout_location="both",
        input_size=(224, 224)
    ):
        super().__init__()
        
        if filter_organization == "same":
            num_filters = [base_filters] * 5
        elif filter_organization == "double":
            num_filters = [base_filters * (2**i) for i in range(5)]
        elif filter_organization == "half":
            num_filters = [base_filters // (2**i) for i in range(5)]
        else:
            num_filters = [base_filters, 64, 128, 256, 512]

        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish()
        }
        self.activation = activation_map[activation]
        
        self.conv_blocks = nn.ModuleList()
        in_channels = 3
        for n_filters in num_filters:
            block = []
            block.append(nn.Conv2d(in_channels, n_filters, 3, padding=1))
            
            if batch_norm:
                block.append(nn.BatchNorm2d(n_filters))
                
            block.append(self.activation)
            
            if dropout_location in ["conv", "both"] and dropout_rate > 0:
                block.append(nn.Dropout2d(dropout_rate))
                
            block.append(nn.MaxPool2d(2, 2))
            
            self.conv_blocks.append(nn.Sequential(*block))
            in_channels = n_filters

        with torch.no_grad():
            dummy = torch.randn(1, 3, *input_size)
            dummy = self.forward_convs(dummy)
            self.flattened_size = dummy.view(1, -1).size(1)

        dense_layers = []
        dense_layers.append(nn.Linear(self.flattened_size, 512))
        
        if dropout_location in ["dense", "both"] and dropout_rate > 0:
            dense_layers.append(nn.Dropout(dropout_rate))
            
        dense_layers.append(activation_map[activation])
        dense_layers.append(nn.Linear(512, num_classes))
        
        self.dense = nn.Sequential(*dense_layers)

    def forward_convs(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_convs(x)
        x = x.view(x.size(0), -1)
        return self.dense(x)