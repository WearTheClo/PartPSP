import torch
import torch.nn as nn


class Com1FrontFFN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.act = nn.Tanh()
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        return x


class Com1BackFFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.act = nn.Tanh()
    
    def forward(self, x):
        res_x = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = res_x + x
        return x


class Com2FrontFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, in_features)
        self.act = nn.Tanh()
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        res_x = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = res_x + x
        return x


class Com2BackFFN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.act = nn.Tanh()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        return x


class FullFFN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, in_features)
        self.linear3 = nn.Linear(in_features, out_features)
        self.act = nn.Tanh()

    def forward(self, x):
        x = torch.flatten(x, 1)
        res_x = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = res_x + x
        x = self.linear3(x)
        x = self.act(x)
        return x

def fused_torch_bottmlp_loader(in_features: int = 1000, num_classes: int = 10, shared: int = 1, pretrained: bool = False):
    if shared == 1:
        ext = Com1FrontFFN(in_features, num_classes)
        loc = Com1BackFFN(num_classes, in_features, num_classes)
        if pretrained:
            ext.load_state_dict(torch.load('./models/Com1FrontFFN.pth', weights_only=True))
            loc.load_state_dict(torch.load('./models/Com1BackFFN.pth', weights_only=True))
        return ext, loc
    elif shared == 2:
        ext = Com2FrontFFN(in_features, num_classes)
        loc = Com2BackFFN(in_features, num_classes)
        if pretrained:
            ext.load_state_dict(torch.load('Com2FrontFFN_old.pth', weights_only=True))
            loc.load_state_dict(torch.load('Com2BackFFN_old.pth', weights_only=True))
        return ext, loc

    #ext = FullFFN(in_features, num_classes)
    #loc = nn.Identity()
    #if pretrained:
    #    ext.load_state_dict(torch.load('./models/FullFFN.pth', weights_only=True))
    #return ext, loc


def unify_torch_bottmlp_loader(in_features: int = 1000, num_classes: int = 10, pretrained: bool = False):
    model = FullFFN(in_features, num_classes)
    if pretrained:
        model.load_state_dict(torch.load('./models/FullFFN.pth', weights_only=True))
    return model

if __name__ == "__main__":
    ext, loc = fused_torch_bottmlp_loader(in_features=784, num_classes=10, shared = 2, pretrained = True)
    #torch.save(ext.state_dict(), 'Com1FrontBottFFN.pth')
    #torch.save(loc.state_dict(), 'Com1BackBottFFN.pth')