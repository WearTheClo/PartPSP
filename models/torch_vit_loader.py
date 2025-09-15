import torchvision.models.vision_transformer as vit
import torch
import torch.nn as nn


class Vit1Front(vit.VisionTransformer):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, num_classes, dropout, attention_dropout):
        super().__init__(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, 
                         hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_classes=num_classes, dropout=dropout, attention_dropout=attention_dropout)

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        return x

class Vit1Back(vit.VisionTransformer):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, num_classes, dropout, attention_dropout):
        super().__init__(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, 
                         hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_classes=num_classes, dropout=dropout, attention_dropout=attention_dropout)

    def forward(self, x):
        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


class Vit2Front(vit.VisionTransformer):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, num_classes, dropout, attention_dropout):
        super().__init__(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, 
                         hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_classes=num_classes, dropout=dropout, attention_dropout=attention_dropout)

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x

class Vit2Back(vit.VisionTransformer):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, num_classes, dropout, attention_dropout):
        super().__init__(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, 
                         hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_classes=num_classes, dropout=dropout, attention_dropout=attention_dropout)

    def forward(self, x):
        x = self.heads(x)

        return x


def fused_shared1_torch_vit_loader(path: str = './models/vit_cifar10_best_layer4_76.pth', num_classes: int = 10, pretrained: bool = False):
    if pretrained:
        checkpoint = torch.load(path, weights_only=True)
        #resized_cp = vit.interpolate_embeddings(image_size=32, patch_size=16, model_state=checkpoint)
        ext_model = Vit1Front(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
        loc_model = Vit1Back(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
        ext_model.load_state_dict(checkpoint)
        loc_model.load_state_dict(checkpoint)
    else:
        ext_model = Vit1Front(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
        loc_model = Vit1Back(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
    #loc_model.heads = torch.nn.Linear(loc_model.heads.head.in_features, num_classes)
    return ext_model, loc_model

def fused_shared2_torch_vit_loader(path: str = './models/vit_cifar10_best_layer4_76.pth', num_classes: int = 10, pretrained: bool = False):
    if pretrained:
        checkpoint = torch.load(path, weights_only=True)
        #resized_cp = vit.interpolate_embeddings(image_size=32, patch_size=16, model_state=checkpoint)
        ext_model = Vit2Front(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
        loc_model = Vit2Back(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
        ext_model.load_state_dict(checkpoint)
        loc_model.load_state_dict(checkpoint)
    else:
        ext_model = Vit2Front(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
        loc_model = Vit2Back(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
    #loc_model.heads = torch.nn.Linear(loc_model.heads.head.in_features, num_classes)
    return ext_model, loc_model


def fused_torch_vit_loader(path: str = './models/vit_cifar10_best_layer4_76.pth', num_classes=10, shared=1, pretrained=True):
    if shared == 1:
        ext_model, loc_model = fused_shared1_torch_vit_loader(path, num_classes, pretrained)
    elif shared == 2:
        ext_model, loc_model = fused_shared2_torch_vit_loader(path, num_classes, pretrained)
    return ext_model, loc_model


def unify_torch_vit_loader(path: str = './models/vit_cifar10_best_layer4_76.pth', num_classes: int = 10, pretrained: bool = False):
    if pretrained:
        checkpoint = torch.load(path, weights_only=True)
        vit_model = vit.VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
        vit_model.load_state_dict(checkpoint)
    else:
        vit_model = vit.VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=1024,
            num_classes=10,
            dropout=0.1,
            attention_dropout=0.1
            )
    return vit_model


if __name__ == "__main__":
    x = torch.rand(64,3,32,32)
    ext, loc = fused_torch_vit_loader(path = './vit_base_p16_224_torch.pth', num_classes=10, shared=1, pretrained=True)
    out = loc(ext(x))
    print(out.size())
    x = torch.rand(64,3,32,32)
    ext, loc = fused_torch_vit_loader(path = './vit_base_p16_224_torch.pth', num_classes=10, shared=2, pretrained=True)
    out = loc(ext(x))
    print(out.size())
    x = torch.rand(64,3,32,32)
    model = unify_torch_vit_loader(path = './vit_base_p16_224_torch.pth', num_classes=10, pretrained=True)
    out = model(x)
    print(out.size())