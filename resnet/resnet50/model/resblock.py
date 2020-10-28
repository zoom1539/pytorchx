import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride = 1):
        super(ResBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace = True),
            nn.Conv2d(channel_out, channel_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(channel_out),
        ) 

        self.shortcut = nn.Sequential()
        if stride != 1 or channel_in != channel_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(channel_out),              
            )
    
    def forward(self, x):
        output = self.stem(x)
        output += self.shortcut(x)
        output = F.relu(output)
        return output