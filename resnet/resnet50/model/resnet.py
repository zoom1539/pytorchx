import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True)
        fc_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(fc_inputs, num_classes)


    def forward(self, x):
        output = self.resnet50(x)

        return output


if __name__=="__main__":
    model = ResNet(num_classes = 3)