from torchvision.datasets import VisionDataset
from torchvision import transforms
import os
from torch.utils.data import Sampler
from collections import defaultdict, Counter
import random
import torch

thermic_transformer = transforms.Compose([
    transforms.Resize((128, 128)),  # a bit more detail preserved
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # simulate noise in temp readings
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize to [-1,1] (optional)
])

# Modified dataset __getitem__ to handle episodic sampling
class ThermicMotorsImagesDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # Get class names and mapping
        self.classes, self.class_to_idx = self._find_classes(root)
        # Collect image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, path):
        """Loads an image from a given path."""
        from PIL import Image
        return Image.open(path).convert("RGB")
    
    def __getitem__(self, idx):
        """
        Modified to handle both normal indexing and episodic indexing
        For episodic sampling, idx can be a tuple (idx, set_type, class_id)
        """
        if isinstance(idx, tuple):
            # Unpack for episodic sampling
            idx, set_type, class_id = idx
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            image = self.load_image(img_path)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label, set_type, class_id
        else:
            # Regular indexing
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            image = self.load_image(img_path)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label   
        
class EpisodicSampler(Sampler):
    def __init__(self, labels, n_way, k_shot, q_query, num_episodes):
        """
        Few-Shot Episodic Sampler
        Args:
            labels (List[int]): List of all labels in dataset (same order as dataset indices)
            n_way (int): Number of classes per episode
            k_shot (int): Number of support samples per class
            q_query (int): Number of query samples per class
            num_episodes (int): How many episodes per epoch
        """
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.class_to_indices = self._build_class_index()
        
        # Verify we have enough classes with sufficient samples
        valid_classes = [cls for cls, indices in self.class_to_indices.items() 
                         if len(indices) >= (self.k_shot + self.q_query)]
        
        print(f"Valid classes for sampling: {len(valid_classes)}")
        print(f"Classes with counts: {[(cls, len(indices)) for cls, indices in self.class_to_indices.items()]}")
        
        if len(valid_classes) < self.n_way:
            raise ValueError(f"Only {len(valid_classes)} classes have enough samples "
                             f"for {self.k_shot}-shot {self.q_query}-query episodes. "
                             f"Cannot create {self.n_way}-way episodes.")
        
        self.valid_classes = valid_classes
    
    def _build_class_index(self):
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def __len__(self):
        return self.num_episodes
    
    def __iter__(self):
        for episode in range(self.num_episodes):
            # Only sample from classes with enough examples
            selected_classes = random.sample(self.valid_classes, self.n_way)
            episode_indices = []
            
            # For each class, clearly separate support and query samples
            for cls in selected_classes:
                # Get available indices for this class
                available_indices = self.class_to_indices[cls]
                
                # Verify we have enough samples for this class
                if len(available_indices) < (self.k_shot + self.q_query):
                    # This should not happen due to our valid_classes check, but just in case
                    raise ValueError(f"Not enough samples ({len(available_indices)}) for class {cls} "
                                    f"in episode {episode}. Need {self.k_shot + self.q_query}.")
                
                # Sample k_shot + q_query samples from this class
                selected_indices = random.sample(available_indices, self.k_shot + self.q_query)
                # The first k_shot indices are for support set
                support_indices = selected_indices[:self.k_shot]
                # The remaining q_query indices are for query set
                query_indices = selected_indices[self.k_shot:]
                
                # Add support indices first, then query indices
                # This way we know exactly which indices belong to which set
                episode_indices.extend((idx, 'support', cls) for idx in support_indices)
                episode_indices.extend((idx, 'query', cls) for idx in query_indices)
            
            yield episode_indices

# Episodic collate function to organize data into support and query sets
def episodic_collate(batch):
    """
    Organize batch data into support and query sets
    Args:
        batch: List of tuples (image, label, set_type, class_id)
    Returns:
        Tuple of (support_images, support_labels, query_images, query_labels)
    """
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    # Maps original class IDs to new contiguous IDs (0 to n_way-1)
    class_mapping = {}
    next_label = 0
    
    for (image, label, set_type, class_id) in batch:
        # Create new label mapping for this episode if needed
        if class_id not in class_mapping:
            class_mapping[class_id] = next_label
            next_label += 1
        
        # Get the episode-specific label (0 to n_way-1)
        episode_label = class_mapping[class_id]
        
        # Add to appropriate set
        if set_type == 'support':
            support_images.append(image)
            support_labels.append(episode_label)
        else:  # query
            query_images.append(image)
            query_labels.append(episode_label)
    
    # Convert to tensors
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)
    
    return support_images, support_labels, query_images, query_labels

# Setup training and testing dataloaders
def setup_fsl_train_test_dataloaders(
    data_path, 
    transform=None, 
    target_transform=None,
    train_classes=None,
    test_classes=None,
    train_ratio=0.7,
    n_way=5, 
    k_shot=1, 
    q_query=5, 
    train_episodes=100, 
    test_episodes=50,
    random_seed=42
):
    """
    Creates separate dataloaders for training and testing
    with disjoint classes to evaluate true few-shot learning.
    
    Args:
        data_path: Path to dataset
        transform: Image transforms
        train_classes: List of class names for training (optional)
        test_classes: List of class names for testing (optional)
        train_ratio: Ratio of classes to use for training if train_classes/test_classes not provided
        n_way, k_shot, q_query: Few-shot learning parameters
        train_episodes: Number of episodes for training
        test_episodes: Number of episodes for testing
        random_seed: For reproducible class splits
    """
    if transform is None:
        transform = thermic_transformer 

    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create full dataset
    full_dataset = ThermicMotorsImagesDataset(
        root=data_path,
        transform=transform,
        target_transform=target_transform
    )
    
    # Count samples per class
    class_counts = defaultdict(int)
    for label in full_dataset.labels:
        class_counts[label] += 1
    
    print(f"Dataset class distribution: {dict(class_counts)}")
    
    # Find classes with enough examples for (k_shot + q_query)
    min_samples_needed = k_shot + q_query
    valid_class_idx = [cls_idx for cls_idx, count in class_counts.items() 
                       if count >= min_samples_needed]
    
    # Map back to class names
    idx_to_class = {idx: cls for cls, idx in full_dataset.class_to_idx.items()}
    valid_classes = [idx_to_class[idx] for idx in valid_class_idx]
    
    print(f"Found {len(valid_classes)} valid classes with at least {min_samples_needed} samples each")
    
    # Check if we have enough classes for n_way classification
    if len(valid_classes) < n_way:
        raise ValueError(f"Only {len(valid_classes)} classes have enough samples. "
                         f"Cannot perform {n_way}-way classification. "
                         f"Reduce n_way or add more data.")
    
    # If classes aren't explicitly provided, split valid classes automatically
    if train_classes is None or test_classes is None:
        # Shuffle classes and split
        shuffled_classes = random.sample(valid_classes, len(valid_classes))
        
        # Ensure we have enough classes for both training and testing
        n_train = max(int(len(shuffled_classes) * train_ratio), n_way)
        
        # Make sure we don't assign more classes than available
        if n_train > len(shuffled_classes) - n_way:
            n_train = len(shuffled_classes) - n_way  # Ensure test gets at least n_way
        
        train_classes = shuffled_classes[:n_train]
        test_classes = shuffled_classes[n_train:n_train + min(len(shuffled_classes) - n_train, len(shuffled_classes))]
        
        # Ensure we have at least n_way classes for testing
        if len(test_classes) < n_way:
            # Move some classes from train to test if needed
            classes_to_move = n_way - len(test_classes)
            test_classes.extend(train_classes[-classes_to_move:])
            train_classes = train_classes[:-classes_to_move]
        
        print(f"Automatically split classes:")
        print(f"  Training classes ({len(train_classes)}): {', '.join(train_classes)}")
        print(f"  Testing classes ({len(test_classes)}): {', '.join(test_classes)}")
    
        # Convert class names to indices
        train_class_indices = [full_dataset.class_to_idx[cls] for cls in train_classes]
        test_class_indices = [full_dataset.class_to_idx[cls] for cls in test_classes]
        
        # Filter indices for training and testing
        train_indices = [i for i, label in enumerate(full_dataset.labels) 
                        if label in train_class_indices]
        test_indices = [i for i, label in enumerate(full_dataset.labels) 
                        if label in test_class_indices]
        
        # Create training dataset with subset of classes
        train_labels = [full_dataset.labels[i] for i in train_indices]
        
        # Create testing dataset with different subset of classes
        test_labels = [full_dataset.labels[i] for i in test_indices]
        
        # Print class distribution in each split
        train_label_counts = Counter(train_labels)
        test_label_counts = Counter(test_labels)
        
        print(f"Training class distribution: {dict(train_label_counts)}")
        print(f"Testing class distribution: {dict(test_label_counts)}")
        
    # Create samplers
    train_n_way = min(n_way, len(train_classes))
    test_n_way = min(n_way, len(test_classes))
    
    print(f"Using n_way={train_n_way} for training, n_way={test_n_way} for testing")
    
    train_sampler = EpisodicSampler(
        labels=train_labels,
        n_way=train_n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=train_episodes
    )
    
    test_sampler = EpisodicSampler(
        labels=test_labels,
        n_way=test_n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=test_episodes   
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_sampler=train_sampler,
        collate_fn=episodic_collate,
        num_workers=os.cpu_count()
    )
    
    test_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_sampler=test_sampler,
        collate_fn=episodic_collate,
        num_workers=os.cpu_count()
    )
    
    return train_loader, test_loader, full_dataset, (train_n_way, test_n_way)


