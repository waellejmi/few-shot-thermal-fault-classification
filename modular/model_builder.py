import torchvision
import torch
# Create a proper feature extractor for Prototypical Networks
class PrototypicalNetwork(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        model_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

        # Setup model
        model = torchvision.models.resnet18(weights=model_weights)
        # Remove the final fully connected layer
        self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
        
        # Add an embedding layer
        self.embedding = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),  
            torch.nn.Linear(in_features=512, out_features=embedding_dim, bias=True) ,
            torch.nn.BatchNorm1d(embedding_dim)  # 128-dim embeddings
            )           
        
        # Optional: freeze some layers of ResNet
        # for param in list(self.encoder.parameters())[:-4]:  # Leave last block trainable
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # # Get features from ResNet
        # x = self.encoder(x)
        # # Flatten
        # x = torch.flatten(x, 1)   # Shape: [batch_size, 512]
        # # Project to embedding space
        # x = self.embedding(x)      # Shape: [batch_size, embedding_dim]
        # # Normalize embeddings (optional but helps with distance calculations)
        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        #WE USED OPERATOR FUSION 
        return torch.nn.functional.normalize(
            self.embedding(
                torch.flatten(self.encoder(x), 1)
                ),
            p=2, dim=1)
   

    
