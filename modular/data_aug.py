import torchvision.transforms as transforms
import torch
import random

# Define custom transform classes that are pickle-compatible
class AddGaussianNoise:
    """Add Gaussian noise to simulate thermal sensor noise"""
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + self.std * torch.randn_like(tensor)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'

class InvertThermalSignature:
    """Occasionally invert thermal signatures to improve robustness"""
    def __init__(self, probability=0.3):
        self.probability = probability
        
    def __call__(self, tensor):
        if random.random() < self.probability:
            return 1.0 - tensor
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + f'(probability={self.probability})'

# Define simple, focused transform for thermal images (training)
thermal_train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),  # Small rotations for alignment variations
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Thermal intensity variations
    transforms.ToTensor(),
    AddGaussianNoise(std=0.01),  # Sensor noise simulation
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# More aggressive augmentation if your model is overfitting
thermal_train_strong_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),  # Trivial Augmentation for wide range of transformations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.05, 0.05),  # Small translations
        scale=(0.95, 1.05),      # Minor scale variations
    ),
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2,
    ),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.015),
    InvertThermalSignature(probability=0.2),  # Occasionally invert thermal image
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),

    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Simple test transform
thermal_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Test-time augmentation transforms (for improved inference)
def get_tta_transforms():
    """
    Returns a list of test-time augmentation transforms.
    Use these at inference time to improve robustness.
    """
    return [
        # Original transform
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        
        # Slight rotation
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        
        # Brightness adjustment
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        
        # Contrast adjustment
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    ]