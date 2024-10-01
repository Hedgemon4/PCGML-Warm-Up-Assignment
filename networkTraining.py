import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from qNetwork import QNetwork

model = QNetwork()

loss_fn = nn.MSELoss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
