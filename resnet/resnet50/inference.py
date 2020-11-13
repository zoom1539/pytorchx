import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary
from torchvision import transforms
import cv2
# from model import ResNet

def main():
    print('cuda device count: ', torch.cuda.device_count())
    # net = ResNet(num_classes = 6)
    net = torch.load('net_401_all.pth')
    net = net.to('cuda:0')
    net.eval()
    print('model: ', net)
    #print('state dict: ', net.state_dict().keys())

    img = cv2.imread('/home/zhumh/code/mag_detect/AOI_resnet/data/test/0.bmp')
    transform = transforms.Compose([
            transforms.ToTensor()  # numpy [H, W, C] [0,255] => tensor [C, H, W] [0.0, 1.0]
            ])
    img_tensor = transform(img)
    print(img_tensor.shape)

    img_tensor = img_tensor.unsqueeze(0)
    print(img_tensor.shape)
    tmp = img_tensor.to('cuda:0')

    # tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    # print('input: ', tmp)

    out = net(tmp)
    print('output:', out)

    summary(net, (3,64,64))
    #return
    f = open("resnet50.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()

