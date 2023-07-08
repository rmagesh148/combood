'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import transforms
from pathlib import Path


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.relu1 = nn.ReLU()
        self.threshold = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y
        #return y, out # changed this for KNN Predict
    
    # def forward_knn(self, x):
    #     out = self.relu1(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     #y = self.linear(out)
    #     y = None
    #     #return y
    #     return y, out # changed this for KNN Predict
    
    def forward_knn(self, x):
        feature1 = self.relu1(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = self.dropout(feature5)
        feature = feature.view(feature.size(0), -1)
        y = None
        #return y
        return y, feature5 # changed this for KNN Predict

    def forward_threshold(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.clip(max=self.threshold)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = self.relu1(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.relu1(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes,
#                                planes,
#                                kernel_size=3,
#                                stride=stride,
#                                padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes,
#                                planes,
#                                kernel_size=3,
#                                stride=1,
#                                padding=1,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes,
#                           self.expansion * planes,
#                           kernel_size=1,
#                           stride=stride,
#                           bias=False), nn.BatchNorm2d(self.expansion * planes))

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes,
#                                planes,
#                                kernel_size=3,
#                                stride=stride,
#                                padding=1,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes,
#                                self.expansion * planes,
#                                kernel_size=1,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes,
#                           self.expansion * planes,
#                           kernel_size=1,
#                           stride=stride,
#                           bias=False), nn.BatchNorm2d(self.expansion * planes))

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block=BasicBlock, num_blocks=None, num_classes=10):
#         super(ResNet, self).__init__()
#         if num_blocks is None:
#             num_blocks = [2, 2, 2, 2]
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3,
#                                64,
#                                kernel_size=3,
#                                stride=1,
#                                padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         # self.avgpool = nn.AvgPool2d(4)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.feature_size = 512 * block.expansion

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
    
#     def forward_knn(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         #y = self.linear(out)
#         y = None
#         #return y
#         return y, out # changed this for KNN Predict

#     def forward(self, x, return_feature=False, return_feature_list=False):
#         feature1 = F.relu(self.bn1(self.conv1(x)))
#         feature2 = self.layer1(feature1)
#         feature3 = self.layer2(feature2)
#         feature4 = self.layer3(feature3)
#         feature5 = self.layer4(feature4)
#         feature5 = self.avgpool(feature5)
#         feature = feature5.view(feature5.size(0), -1)
#         logits_cls = self.fc(feature)
#         feature_list = [feature1, feature2, feature3, feature4, feature5]
#         if return_feature:
#             return logits_cls, feature
#         elif return_feature_list:
#             return logits_cls, feature_list
#         else:
#             return logits_cls

#     def forward_threshold(self, x, threshold):
#         feature1 = F.relu(self.bn1(self.conv1(x)))
#         feature2 = self.layer1(feature1)
#         feature3 = self.layer2(feature2)
#         feature4 = self.layer3(feature3)
#         feature5 = self.layer4(feature4)
#         feature5 = self.avgpool(feature5)
#         feature = feature5.clip(max=threshold)
#         feature = feature.view(feature.size(0), -1)
#         logits_cls = self.fc(feature)

#         return logits_cls
    
#     # function to extact the multiple features
#     def feature_list(self, x):
#         out_list = []
#         out = F.relu(self.bn1(self.conv1(x)))
#         out_list.append(out)
#         out = self.layer1(out)
#         out_list.append(out)
#         out = self.layer2(out)
#         out_list.append(out)
#         out = self.layer3(out)
#         out_list.append(out)
#         out = self.layer4(out)
#         out_list.append(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         return y, out_list

#     # function to extact a specific feature
#     def intermediate_forward(self, x, layer_index):
#         out = F.relu(self.bn1(self.conv1(x)))
#         if layer_index == 1:
#             out = self.layer1(out)
#         elif layer_index == 2:
#             out = self.layer1(out)
#             out = self.layer2(out)
#         elif layer_index == 3:
#             out = self.layer1(out)
#             out = self.layer2(out)
#             out = self.layer3(out)
#         elif layer_index == 4:
#             out = self.layer1(out)
#             out = self.layer2(out)
#             out = self.layer3(out)
#             out = self.layer4(out)
#         return out

#     # function to extact the penultimate features
#     def penultimate_forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         penultimate = self.layer4(out)
#         out = F.avg_pool2d(penultimate, 4)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         return y, penultimate

#     # def get_fc(self):
#     #     fc = self.fc
#     #     return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()


def ResNet18(num_c):
    print("Number of classes: ", num_c)
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_c) # added this for OpenOOD
    #return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_c)


def ResNet34(num_c):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_c)


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())


def save_full_model(path, name):
    model = ResNet34(num_c=10)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path / name, map_location=device))
    model.eval()
    torch.save(model, path / "model.pt")
    # in_transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])


if __name__ == "__main__":
    save_full_model(Path("svhn/resnet"), "resnet_svhn.pth")
