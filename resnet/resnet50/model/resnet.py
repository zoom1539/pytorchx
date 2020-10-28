import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True)
        fc_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
)

    def forward(self, x):
        output = self.resnet50(x)

        return output


if __name__=="__main__":
    model = ResNet(num_classes = 3)