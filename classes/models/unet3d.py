import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, channel_selection: int = 0, double_conv_kernel_size:int =3, is_upsampling = False):
        super(UNet3D, self).__init__()
        _channels = ()
        if channel_selection == 0:
            _channels = (4, 8, 16, 32, 64)
        elif channel_selection == 1:
            _channels = (8, 16, 32, 64, 128)
        elif channel_selection == 2:
            _channels = (16, 32, 64, 128, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.convtype = nn.Conv3d
        
        self.is_upsampling = is_upsampling

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype, kernel_size=double_conv_kernel_size)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down4 = Down(self.channels[3], self.channels[4], conv_type=self.convtype)
        
        self.up1 = Up(self.channels[4], self.channels[3], is_upsampling)
        self.up2 = Up(self.channels[3], self.channels[2], is_upsampling)
        self.up3 = Up(self.channels[2], self.channels[1], is_upsampling)
        self.up4 = Up(self.channels[1], self.channels[0], is_upsampling)

        
        self.outc = OutConv(self.channels[0], n_classes)
         
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None, kernel_size: int=3):
        super().__init__()
        padding = 1
        if kernel_size == 5:
            padding = 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, is_upsampling = False):
        super().__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.is_upsampling = is_upsampling
        
        if is_upsampling:
            self.up = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
            self.conv3d = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
            

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.is_upsampling:
            x1 = self.conv3d(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
