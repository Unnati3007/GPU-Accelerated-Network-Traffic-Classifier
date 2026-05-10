import torch
import torch.nn as nn


class TrafficClassifier(nn.Module):
    """
    Feed-forward DNN for network traffic classification.
    Input: 78 flow-level features from CICIDS2017
    Output: 5-class softmax (Benign, DDoS, PortScan, BruteForce, WebAttack)
    """

    def __init__(self, input_dim: int = 78, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
