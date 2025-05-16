import torch
import torch.nn as nn

class MobileNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1, **kwargs):
        super(MobileNetV2Block, self).__init__()
        self.expansion_factor = expansion_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_channels = in_channels * expansion_factor
        
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.expansion_channels,
            kernel_size=1,
            bias=False
        )

        self.act1 = nn.ReLU6()

        self.bn1 = nn.BatchNorm2d(self.expansion_channels)

        self.d1 = nn.Conv2d(
            in_channels=self.expansion_channels,
            out_channels=self.expansion_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.expansion_channels,
            bias=False
        )

        self.act2 = nn.ReLU6()

        self.bn2 = nn.BatchNorm2d(self.expansion_channels)

        self.c2= nn.Conv2d(
            in_channels=self.expansion_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(self.out_channels)

        self.residual = (self.stride == 1 and self.in_channels == self.out_channels)

    def forward(self, x):
        out = x
        x = self.c1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.d1(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.c2(x)
        x = self.bn3(x)

        if self.residual:
            x += out

        return x                

class MobileNetV2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, n, **kwargs):
        super(MobileNetV2Layer, self).__init__()
        layers = [MobileNetV2Block(in_channels, out_channels, **kwargs)] + [MobileNetV2Block(out_channels, out_channels, **kwargs) for _ in range(n-1)]

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x        

class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, 
                 classes=1000, 
                 expansion_factor_list = [-1,1,6,6,6,6,6,6], 
                 out_channel_list = [32,16,24,32,64,96,160,320,1280], 
                 layer_list = [1,1,2,3,4,3,3,1], 
                 stride_list = [2,1,2,2,2,1,2,1]):
        
        super(MobileNetV2, self).__init__()
        
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channel_list[0],
            kernel_size=3,
            stride=stride_list[0],
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel_list[0])
        self.act1 = nn.ReLU6()
        
        layers = []
        for i in range(1, len(expansion_factor_list)):
            expansion_factor = expansion_factor_list[i]
            in_channels = out_channel_list[i-1]
            out_channels = out_channel_list[i]
            n = layer_list[i]
            stride = stride_list[i]

            layers.append(MobileNetV2Layer(in_channels, 
                                           out_channels, 
                                           n,
                                           expansion_factor=expansion_factor,
                                           stride=stride))

        self.model = nn.Sequential(*layers)

        self.c2 = nn.Conv2d(
            in_channels=out_channel_list[-2],
            out_channels=out_channel_list[-1],
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel_list[-1])
        self.act2 = nn.ReLU6()

        self.gap = nn.AdaptiveAvgPool2d(1)

        # self.c3 = nn.Conv2d(
        #     in_channels=out_channel_list[-1],
        #     out_channels=classes,
        #     kernel_size=1,
        #     bias=False
        # )

    def forward(self, x):
        x = self.c1(x)
        x = self.act1(x)
        x = self.bn1(x)

        x = self.model(x)

        x = self.c2(x)
        x = self.act2(x)
        x = self.bn2(x)

        x = self.gap(x)

        # x = self.c3(x)
        return x


class myLayerNorm(nn.Module):
  def __init__(self, dims, eps=1e-6):
    super().__init__()
    self.norm = nn.LayerNorm(dims, eps)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    x = self.norm(x)
    x = x.permute(0, 3, 1, 2)

    return x

class ConvNeXtBlock2D(nn.Module):
  def __init__(self, dim, layer_scale_init_value=1e-6, drop=0.2):
    super().__init__()

    self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv

    self.norm = nn.LayerNorm(dim)

    self.pwconv1 = nn.Linear(dim, 4 * dim)
    self.act = nn.GELU()
    self.pwconv2 = nn.Linear(4 * dim, dim)

    self.dropout = nn.Dropout2d(p=drop)

    self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None


  def forward(self, x):
    residual = x
    x = self.dwconv(x)

    # Transpose for LayerNorm
    x = x.permute(0, 2, 3, 1)
    x = self.norm(x)

    x = self.pwconv1(x)
    x = self.act(x)
    x = self.pwconv2(x)

    x = self.dropout(x)

    if self.gamma is not None:
        x = self.gamma * x

    # Transpose back to (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    # no drop path y
    return residual + x

class ConvNext(nn.Module):
    def __init__(self, in_chans=1, dims=[32, 64, 128, 256], stages=[1, 1, 3, 1]):
        super().__init__()

        self.in_chans = in_chans
        self.dims = dims
        self.stages = stages

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            myLayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                myLayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.model_layers = nn.ModuleList()
        for i, stage_length in enumerate(stages):
            stage = nn.ModuleList([ConvNeXtBlock2D(dims[i]) for _ in range(stage_length)])
            self.model_layers.append(stage)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.final_norm = nn.LayerNorm(dims[-1])

    def forward(self, x):
        for i in range(len(self.dims)):
            x = self.downsample_layers[i](x)
            for layer in self.model_layers[i]:
                x = layer(x)

        x = self.pooling(x)
        x = self.flatten(x)
        x = self.final_norm(x)  # Final normalization
        return x

class ClassificationTail(nn.Module):
    def __init__(self, in_features, num_classes=2, dropout_rate=0.5):
        super(ClassificationTail, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(in_features // 2, in_features // 4),
            nn.BatchNorm1d(in_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(in_features // 4, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class RegressionTail(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5):
        super().__init__()
        self.fc = nn.Sequential(
             nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(in_features // 2, in_features // 4),
            nn.BatchNorm1d(in_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
        )
        self.out = nn.Linear(in_features // 4, 1)
    def forward(self,x):
        x = self.fc(x)
        x = self.out(x)
        return x

class FullModel(nn.Module):
    def __init__(self, backbone, tail):
        super(FullModel, self).__init__()
        self.backbone = backbone
        self.tail = tail

    def forward(self, x):
        x = self.backbone(x)
        x = self.tail(x)
        return x