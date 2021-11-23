import torch
import torchvision
import torch.nn as nn

class Spatial_Loss(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Loss, self).__init__()
        self.res_scale = in_channels
        
        self.make_PAN = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

        self.L1_loss = nn.L1Loss()
        
    def forward(self, ref_HS, pred_HS):
        pan_pred = self.make_PAN(pred_HS)
        with torch.no_grad():
            pan_ref = self.make_PAN(ref_HS)     
        spatial_loss = self.L1_loss(pan_pred, pan_ref.detach())
        return spatial_loss