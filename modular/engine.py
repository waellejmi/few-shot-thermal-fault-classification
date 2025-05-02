from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch

from modular.utils import save_model

def prototypical_loss(query_embeddings, support_embeddings, query_labels, support_labels, n_way):
    """
    Calculates the prototypical loss as described in the paper.
    
    Args:
        query_embeddings: Tensor of shape [n_way*q_query, embedding_dim]
        support_embeddings: Tensor of shape [n_way*k_shot, embedding_dim]
        query_labels: Tensor of shape [n_way*q_query]
        support_labels: Tensor of shape [n_way*k_shot]
        n_way: Number of classes per episode
    """
    # Input validation
    assert support_embeddings.shape[0] >= n_way, f"Not enough support samples ({support_embeddings.shape[0]}) for {n_way}-way classification"
    assert query_embeddings.shape[1] == support_embeddings.shape[1], "Embedding dimensions don't match"
    
    # Calculate prototypes more efficiently
    prototypes = torch.stack([
        support_embeddings[support_labels == c].mean(0)
        for c in range(n_way)
    ])
    
    # Compute distances
    dists = torch.cdist(query_embeddings, prototypes, p=2)
    
    # Temperature scaling for better gradient flow (optional)
    temperature = 0.1
    logits = -dists / temperature
    
    # Compute predictions and accuracy
    with torch.inference_mode():
        preds = torch.argmin(dists, dim=1)
        accuracy = (preds == query_labels).float().mean().item()
    
    # Compute loss with label smoothing (optional)
    label_smoothing = 0.1
    loss = torch.nn.functional.cross_entropy(
        logits, 
        query_labels,
        label_smoothing=label_smoothing
    )
    
    return loss, accuracy


def train_step(model, dataloader, optimizer, n_way,device):
    """Train step for prototypical networks."""
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    # Add gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    

    

    for batch_idx, (support_images, support_labels, query_images, query_labels) in enumerate(dataloader):
        # # Move to device
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)
        
        # Get embeddings
        support_embeddings = model(support_images)
        query_embeddings = model(query_images)
        
        # Calculate loss and accuracy
        loss, accuracy = prototypical_loss(
            query_embeddings, 
            support_embeddings, 
            query_labels, 
            support_labels, 
            n_way
        )
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        epoch_acc += accuracy


    
     # Average metrics
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    epoch_acc /= num_batches
    
    return epoch_loss, epoch_acc
    

def test_step(model, dataloader, n_way,device):
    """Test/evaluation step for prototypical networks."""
    model.eval()
    test_loss = 0
    test_acc = 0
    
    # Use torch.no_grad to disable gradient calculation during inference
    with torch.inference_mode():
        
        for batch_idx, (support_images, support_labels, query_images, query_labels) in  enumerate(dataloader):
            
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            # Get embeddings
            support_embeddings = model(support_images)
            query_embeddings = model(query_images)
            
            # Calculate loss and accuracy using the same prototypical_loss function
            loss, accuracy = prototypical_loss(
                query_embeddings, 
                support_embeddings, 
                query_labels, 
                support_labels, 
                n_way
            )
            
            # Update metrics
            test_loss += loss.item()
            test_acc += accuracy
            
    
      
    
    # Average metrics
    num_batches = len(dataloader)
    test_loss /= num_batches
    test_acc /= num_batches
    
    return test_loss, test_acc

def train_prototype_network(
    model, 
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    n_way,
    device,
    epochs, 
    model_name=f"ResNet18.pth",
    start_epoch=0,            
    writer=SummaryWriter(),  # Optional: TensorBoard writer
    patience=14,  # For early stopping
     
):
    """
    Training function for Prototypical Network with testing
    and early stopping capabilities.
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # For early stopping    
    best_test_acc = 0
    counter = 0
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

    epoch_pbar = tqdm(range(start_epoch,epochs), desc="Training Progress")
    # Loop through training for the number of epochs
    for epoch in epoch_pbar:
        
        # Training step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            n_way=n_way,
            device=device
        )
        
        # Testing step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            n_way=n_way,
            device=device
        )
      
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            # Print results
            print(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )
            
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        scheduler.step()
        ## SAVING BEST MODEL WILL FIX LATER    
        # Check for overfitting and early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            counter = 0
            # SAVE THE NEW BEST MODEL HERE
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                target_dir=r"C:\Users\Wael\Code\FSL\models",
                model_name=model_name
            )
            print(f"[INFO] New best model (acc={test_acc:.4f}) saved at epoch {epoch+1}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    if writer:
        # Add results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"train_acc": train_acc,
                                            "test_acc": test_acc}, 
                           global_step=epoch)
        # Close the writer
        writer.close()
    else:
        pass

    return results