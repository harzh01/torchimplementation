import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

from train import train_large,train_distil
from evaluate import eval
from model import large,distil

def main():
    batch_size = 100

    # Loading MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'lr': [0.01, 0.05],
        'momentum': [0.9, 0.95],
        'epochs': [10, 15],
    }

    print("Starting Grid Search for Large Model...")
    best_params, best_accuracy = grid_search_hyperparams(train_loader, test_loader, param_grid)

    # Train Distil Model with best parameters found
    large_model = LargeModel().to(device)
    optimizer_large = optim.SGD(large_model.parameters(), lr=best_params['lr'], momentum=best_params['momentum'])
    print("Training Large Model with Best Parameters...")
    train_large(large_model, train_loader, optimizer_large, best_params['epochs'], device)

    print("Evaluating Large Model with Best Parameters...")
    evaluate(large_model, test_loader, device)

    distil_model = DistilModel().to(device)
    optimizer_distil = optim.SGD(distil_model.parameters(), lr=best_params['lr'], momentum=best_params['momentum'])
    loss_fn = nn.KLDivLoss(reduction='batchmean')

    print("Training Distil Model...")
    train_distil(
        large_model=large_model,
        distil_model=distil_model,
        train_loader=train_loader,
        optimizer=optimizer_distil,
        loss_fn=loss_fn,
        device=device,
        epochs=best_params['epochs'],
        temp=20,
        distil_weight=0.7
    )

    print("Evaluating Distil Model...")
    evaluate(distil_model, test_loader, device)

if __name__ == '__main__':
    main()