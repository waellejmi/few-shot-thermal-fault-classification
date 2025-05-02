import torch
from torchvision import transforms
from pathlib import Path

import model_builder, data_setup, engine, utils

def main():
    # Hyperparameters
    EPOCHS = 3
    TRAIN_EPISODES = 200
    TEST_EPISDOES = 200
    N_WAY = 5
    K_SHOT = 2
    Q_QUERY = 5
    LEARNING_RATE = 0.001

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    data_path = Path(r"C:\Users\Wael\Code\FSL\data")

    # Data transforms
    thermic_transformer = transforms.Compose([
    transforms.Resize((128, 128)),  # a bit more detail preserved
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # simulate noise in temp readings
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize to [-1,1] (optional)
])


    # Data loaders
    train_loader, test_loader, _, _ = data_setup.setup_fsl_train_test_dataloaders(
        data_path=data_path,
        transform=thermic_transformer,
        train_episodes=TRAIN_EPISODES,
        test_episodes=TEST_EPISDOES,
        n_way=N_WAY,
        k_shot=K_SHOT,
        q_query=Q_QUERY,
    )

    # Model
    model = model_builder.PrototypicalNetwork(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    results = engine.train_prototype_network(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        n_way=N_WAY,
        device=device,
        epochs=EPOCHS,
        patience=20
    )

    # Results
    utils.plot_training_results(results=results)
    utils.analyze_model_performance(results=results)
    utils.save_model(model=model, model_name="prototypical_network", save_path=Path("models/scripts/"))

if __name__ == "__main__":
    main()
