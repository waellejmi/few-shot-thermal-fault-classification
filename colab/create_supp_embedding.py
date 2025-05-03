import torch
from torchvision import transforms
from PIL import Image
import torchvision
import os


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



# Configuration
DATA_DIR = r"C:\Users\Wael\Code\FSL\data"  # Your local dataset path
MODEL_PATH = "Resnet18_RetrainedV2.pth"
CLASS_NAMES = [
    'A&B50', 'A&C&B10', 'A&C&B30', 'A&C10', 'A&C30',
    'A10', 'A30', 'A50', 'Fan', 'Noload', 'Rotor-0'
]

K_SHOT = 5

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Resnet18(embedding_dim=256)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

# Generate and save embeddings
support_embeddings = []
for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(DATA_DIR, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png','.bmp'))][:K_SHOT]
    
    class_embs = []
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img)
        class_embs.append(emb)
    
    support_embeddings.append(torch.mean(torch.cat(class_embs), dim=0))

torch.save(torch.stack(support_embeddings), "support_embeddings.pt")
print("Embeddings saved!")