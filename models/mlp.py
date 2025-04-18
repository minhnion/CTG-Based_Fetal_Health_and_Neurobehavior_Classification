import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMultiOutput(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10, num_nsp: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.2)

        self.class_head = nn.Linear(64, num_classes)
        self.nsp_head   = nn.Linear(64, num_nsp)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        # raw logits
        return self.class_head(x), self.nsp_head(x)
