import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Models
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1600)  # Increased hidden units
        self.fc2 = nn.Linear(1600, 1600)  # Increased hidden units
        self.fc3 = nn.Linear(1600, 10)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return out

class DistilModel(nn.Module):
    def __init__(self):
        super(DistilModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 800)  # Increased hidden units slightly
        self.fc2 = nn.Linear(800, 800)  # Increased hidden units slightly
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out