def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = len(data_loader.dataset)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f} ({correct}/{total})')
    return accuracy