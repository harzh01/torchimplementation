import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Train Large Model
def train_large(model, train_loader, optimizer, epochs, device):
    model.train()
    loss_arr = []

    for e in range(epochs):
        epoch_loss = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_arr.append(epoch_loss)
        print(f'Epoch {e+1}/{epochs}, Loss: {epoch_loss:.4f}')

    plt.plot(loss_arr)
    plt.title('Training Loss - Large Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Train Distil Model
def train_distil(large_model, distil_model, train_loader, optimizer, loss_fn, device, epochs=10, temp=20, distil_weight=0.7):
    large_model.eval()
    distil_model.train()
    loss_arr = []

    for e in range(epochs):
        epoch_loss = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                soft_label = F.softmax(large_model(data) / temp, dim=1)

            out = distil_model(data)
            soft_out = F.softmax(out / temp, dim=1)

            loss = (1 - distil_weight) * F.cross_entropy(out, label) + distil_weight * loss_fn(soft_out, soft_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_arr.append(epoch_loss)
        print(f'Epoch {e+1}/{epochs}, Loss: {epoch_loss:.4f}')

    plt.plot(loss_arr)
    plt.title('Training Loss - Distil Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def grid_search_hyperparams(train_loader, test_loader, param_grid):
    best_params = None
    best_accuracy = 0.0

    for params in ParameterGrid(param_grid):
        print(f"Testing with parameters: {params}")

        large_model = LargeModel().to(device)
        optimizer_large = optim.SGD(large_model.parameters(), lr=params['lr'], momentum=params['momentum'])

        print("Training Large Model...")
        train_large(large_model, train_loader, optimizer_large, params['epochs'], device)

        accuracy = evaluate(large_model, test_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    return best_params, best_accuracy