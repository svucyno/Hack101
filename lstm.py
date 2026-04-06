import torch
import torch.nn as nn
import numpy as np

class LSTMAccidentDetector(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.softmax(out)

# Features per frame: [conf_accident, conf_fire, conf_normal, count_accident, count_fire, count_normal]
# Sequence: last 10 frames

# Usage example (train separately if data available)
# model = LSTMAccidentDetector()
# Train with sequences labeled accident/fire/normal
# For now, use pretrain or dummy

# Dummy trained model load (user to train/replace)
# torch.save(model.state_dict(), 'lstm_accident.pt')

