import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import os

from modular import model_builder

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)
K_SHOT = 5  # Number of support examples per class
MODEL_PATH = Path("models/Resnet18_RetrainedV2.pth")
DATA_DIR = "data"  # Your data directory with class folders

# All 11 classes in order
CLASS_NAMES = [
    'A&B50', 'A&C&B10', 'A&C&B30', 'A&C10', 'A&C30',
    'A10', 'A30', 'A50', 'Fan', 'Noload', 'Rotor-0'
]

def load_model():
    """Load trained model with embeddings"""
    model = model_builder.Resnet18(embedding_dim=256)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model

def get_fixed_support_set(data_dir, k_shot):
    """
    Create consistent support set with all 11 classes
    Returns:
        support_images: Tensor [11*k_shot, C, H, W]
        support_labels: Tensor [11*k_shot]
        CLASS_NAMES: The fixed class order
    """
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    support_images = []
    support_labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        try:
            # Get all available images in class directory
            available_images = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # If not enough images, use all available with warning
            if len(available_images) < k_shot:
                print(f"Warning: Class {class_name} has only {len(available_images)} images")
                selected_images = available_images
            else:
                selected_images = random.sample(available_images, k_shot)
            
            # Load and transform images
            for img_name in selected_images:
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                support_images.append(transform(img))
                support_labels.append(class_idx)
                
        except FileNotFoundError:
            print(f"Error: Class directory not found - {class_dir}")
            continue
    
    if not support_images:
        raise ValueError("No valid support images found. Check your data directory.")
    
    return (torch.stack(support_images).to(DEVICE),
            torch.tensor(support_labels).to(DEVICE),
            CLASS_NAMES)

def predict_single_image(model, image_path, support_set):
    """
    Predict class for a single query image using all 11 classes
    
    Args:
        model: Your trained model
        image_path: Path to query image
        support_set: Tuple from get_fixed_support_set()
    
    Returns:
        pred_class: Predicted class name
        confidence: Softmax probability
        distances: Raw distances to all prototypes
    """
    support_images, support_labels, class_names = support_set
    
    # Transform for query image
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        query_img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None
    
    with torch.no_grad():
        # Get embeddings
        support_embeddings = model(support_images)
        query_embedding = model(query_img)
        
        # Calculate prototypes (mean of support embeddings per class)
        prototypes = torch.stack([
            support_embeddings[support_labels == c].mean(0)
            for c in range(len(class_names))
        ])
        
        # Euclidean distances
        distances = torch.cdist(query_embedding, prototypes, p=2).squeeze()
        
        # Convert distances to probabilities with temperature scaling
        temperature = 0.1  # Lower = more confident predictions
        probs = torch.softmax(-distances/temperature, dim=0)
        pred_idx = torch.argmax(probs).item()
        
    return class_names[pred_idx], probs[pred_idx].item(), distances.cpu().numpy()

def main():
    print("Loading model...")
    model = load_model()
    
    print("Creating fixed support set with all 11 classes...")
    try:
        support_set = get_fixed_support_set(DATA_DIR, K_SHOT)
    except ValueError as e:
        print(f"Error creating support set: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        query_path = input("Enter path to query image (or 'quit' to exit): ").strip('"\' ')
        
        if query_path.lower() in ['quit', 'exit', 'q']:
            break
            
        if not os.path.isfile(query_path):
            print(f"File not found: {query_path}")
            continue
            
        pred_class, confidence, distances = predict_single_image(model, query_path, support_set)
        
        if pred_class is None:
            continue
            
        print("\n--- Prediction Results ---")
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {confidence:.2%}")
        
        print("\nDistances to all prototypes:")
        for cls, dist in zip(CLASS_NAMES, distances):
            print(f"  {cls:<8}: {dist:.4f}")

if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducible support sampling
    main()