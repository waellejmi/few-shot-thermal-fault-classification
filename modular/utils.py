
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import random
from typing import Tuple, Dict, List
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
  
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    if seed:
        random.seed(seed)
    
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        
        # Denormalize the image
        targ_image = targ_image * 0.5 + 0.5
        
        # Adjust tensor shape for plotting
        targ_image_adjust = targ_image.permute(1, 2, 0).numpy()
        
        # Handle grayscale images
        if targ_image_adjust.shape[-1] == 1:
            targ_image_adjust = targ_image_adjust.squeeze(-1)
        
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust, cmap='gray' if targ_image_adjust.ndim == 2 else None)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title += f"\nshape: {targ_image_adjust.shape}"
            plt.title(title)
        
def plot_training_results(results):
    """
    Plot training and testing curves.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training & testing loss
    plt.subplot(1, 2, 1)
    plt.plot(results["train_loss"], label="Training Loss")
    plt.plot(results["test_loss"], label="Testing Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot training & testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results["train_acc"], label="Training Accuracy")
    plt.plot(results["test_acc"], label="Testing Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()



def create_writer(experiment_name: str, model_name: str, extra: str = None):
    """Creates a SummaryWriter instance saving to a specific log_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d")  # Current date in YYYY-MM-DD format
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path) 

def analyze_model_performance(results):
    """
    Analyze model performance and detect potential overfitting.
    """
    # Get final metrics
    final_train_loss = results["train_loss"][-1]
    final_test_loss = results["test_loss"][-1]
    final_train_acc = results["train_acc"][-1]
    final_test_acc = results["test_acc"][-1]
    
    # Calculate the gap between training and testing
    loss_gap = final_train_loss - final_test_loss
    acc_gap = final_train_acc - final_test_acc
    
    print(f"Final Training Loss: {final_train_loss:.4f}, Final Testing Loss: {final_test_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.4f}, Final Testing Accuracy: {final_test_acc:.4f}")
    
    # Check for overfitting
    if acc_gap > 0.15:  # If training accuracy is much higher than testing
        print("WARNING: Possible overfitting detected. Training accuracy is significantly higher than testing accuracy.")
    elif final_test_loss > 1.5 * final_train_loss:
        print("WARNING: Possible overfitting detected. Testing loss is significantly higher than training loss.")
    else:
        print("Model appears to be generalizing well with minimal overfitting.")
        
    # Get best testing accuracy and its epoch
    best_epoch = results["test_acc"].index(max(results["test_acc"])) + 1
    best_acc = max(results["test_acc"])
    print(f"Best testing accuracy {best_acc:.4f} achieved at epoch {best_epoch}")
    
    return {
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "final__loss_gap": loss_gap,
        "final__acc_gap": acc_gap,
        "overfitting_detected": acc_gap > 0.15 or final_test_loss > 1.5 * final_train_loss
    }