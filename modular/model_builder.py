import torchvision
import torch
# Create a proper feature extractor for Prototypical Networks
class Resnet18(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        model_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1        # Setup model
        model = torchvision.models.resnet18(weights=model_weights)
        self.encoder = torch.nn.Sequential(*list(model.children())[:-1])

        self.embedding = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),  
            torch.nn.Linear(in_features=512, out_features=embedding_dim, bias=True) ,
            torch.nn.BatchNorm1d(embedding_dim)  # 128-dim embeddings
            )           
        
        for param in list(self.encoder.parameters())[:-4]:  
            param.requires_grad = False
            
    def forward(self, x):
        return torch.nn.functional.normalize(
        self.embedding(torch.flatten(self.encoder(x), 1)),
        p=2, dim=1
        )


class MobileNetV2(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        model_weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
        model = torchvision.models.mobilenet_v2(weights=model_weights)

        self.encoder = model.features  # Output: [B, 1280, 4, 4] for 128x128 input
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Make it [B, 1280, 1, 1]
        
        self.embedding = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=embedding_dim, bias=True),
            torch.nn.BatchNorm1d(embedding_dim),
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)            # [B, 1280, H, W]
        x = self.pool(x)               # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)        # [B, 1280]
        x = self.embedding(x)          # [B, embedding_dim]
        return torch.nn.functional.normalize(x, p=2, dim=1)
