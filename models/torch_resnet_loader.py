import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class ResAll(resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Res1Front(resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        return x

class Res1Back(resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def _forward_impl(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Res2Front(resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Res2Back(resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def _forward_impl(self, x):
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Res3Front(resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Res3Back(resnet.ResNet):
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def _forward_impl(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def Allresnet18(*, weights, progress: bool = True):
    model = ResAll(resnet.BasicBlock, [2, 2, 2, 2], num_classes = 1000)
    if weights is not None:
        weights = resnet.ResNet18_Weights.verify(weights)
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

def Fuseresnet18(*, weights, progress: bool = True, shared: int = 1):
    if shared == 1:
        ext_model = Res1Front(resnet.BasicBlock, [2, 2, 2, 2], num_classes = 1000)
        cls_model = Res1Back(resnet.BasicBlock, [2, 2, 2, 2], num_classes = 1000)
    elif shared == 2:
        ext_model = Res2Front(resnet.BasicBlock, [2, 2, 2, 2], num_classes = 1000)
        cls_model = Res2Back(resnet.BasicBlock, [2, 2, 2, 2], num_classes = 1000)
    else:
        ext_model = Res3Front(resnet.BasicBlock, [2, 2, 2, 2], num_classes = 1000)
        cls_model = Res3Back(resnet.BasicBlock, [2, 2, 2, 2], num_classes = 1000)

    if weights is not None:
        weights = resnet.ResNet18_Weights.verify(weights)
        ext_model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        cls_model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return ext_model, cls_model

def Fuseresnet34(*, weights, progress: bool = True, shared: int = 1):
    if shared == 1:
        ext_model = Res1Front(resnet.BasicBlock, [3, 4, 6, 3], num_classes = 1000)
        cls_model = Res1Back(resnet.BasicBlock, [3, 4, 6, 3], num_classes = 1000)
    elif shared == 2:
        ext_model = Res2Front(resnet.BasicBlock, [3, 4, 6, 3], num_classes = 1000)
        cls_model = Res2Back(resnet.BasicBlock, [3, 4, 6, 3], num_classes = 1000)
    else:
        ext_model = Res3Front(resnet.BasicBlock, [3, 4, 6, 3], num_classes = 1000)
        cls_model = Res3Back(resnet.BasicBlock, [3, 4, 6, 3], num_classes = 1000)

    if weights is not None:
        weights = resnet.ResNet34_Weights.verify(weights)
        ext_model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        cls_model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return ext_model, cls_model


def fused_torch_resnet18_loader(num_classes: int = 10, pretrained: bool = False, shared: int = 1):
    if pretrained:
        ext_model, cls_model = Fuseresnet18(weights = resnet.ResNet18_Weights.IMAGENET1K_V1, shared = shared)
    else:
        ext_model, cls_model = Fuseresnet18(weights = None, shared = shared)

    cls_model.fc = nn.Linear(cls_model.fc.in_features, num_classes)
    return ext_model, cls_model

def fused_torch_resnet34_loader(num_classes: int = 10, pretrained: bool = False, shared: int = 1):
    if pretrained:
        ext_model, cls_model = Fuseresnet34(weights = resnet.ResNet34_Weights.IMAGENET1K_V1, shared = shared)
    else:
        ext_model, cls_model = Fuseresnet34(weights = None, shared = shared)

    cls_model.fc = nn.Linear(cls_model.fc.in_features, num_classes)
    return ext_model, cls_model

def unify_torch_resnet_loader(model: str, num_classes: int = 10, pretrained: bool = False):
    if pretrained:
        res_model = Allresnet18(weights = resnet.ResNet18_Weights.IMAGENET1K_V1)
    else:
        res_model = Allresnet18(weights = None)

    res_model.fc = nn.Linear(res_model.fc.in_features, num_classes)
    return res_model

if __name__ == "__main__":
    ext18, loc18 = fused_torch_resnet18_loader(num_classes = 10, pretrained = False, shared = 3)
    ext18pre, loc18pre = fused_torch_resnet18_loader(num_classes = 10, pretrained = True, shared = 3)
    ext34, loc34 = fused_torch_resnet34_loader(num_classes = 10, pretrained = False, shared = 3)
    ext34pre, loc34pre = fused_torch_resnet34_loader(num_classes = 10, pretrained = True, shared = 3)

    x = torch.rand(10, 3, 32, 32)
    mid18 = ext18(x)
    y = loc18(mid18)
    print("ext18 good")

    mid18 = ext18pre(x)
    y = loc18pre(mid18)
    print("ext18 good")

    mid18 = ext34(x)
    y = loc34(mid18)
    print("ext18 good")

    mid18 = ext34pre(x)
    y = loc34pre(mid18)
    print("ext18 good")
