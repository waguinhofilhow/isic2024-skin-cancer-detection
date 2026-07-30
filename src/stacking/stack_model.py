import torch.nn as nn


class StackingModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(3, 4),

            nn.ReLU(),

            nn.Linear(4, 1),

        )

    def forward(self, x):

        return self.network(x).squeeze(1)