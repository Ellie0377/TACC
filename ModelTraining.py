import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, conv_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, conv_channels: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional, # BiLSTM，Encoder 和 Decoder 的方向性需一樣，否則維度會報錯
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, conv_channels: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional, # BiLSTM
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # BiLSTM 的 hidden size 會翻倍 
        direction_factor = 2 if bidirectional else 1
        self.output_layer = nn.Linear(hidden_size * direction_factor, conv_channels)

    def forward(self, decoder_input, hidden, cell):
        decoded, _ = self.lstm(decoder_input, (hidden, cell))
        return self.output_layer(decoded)


class ReconstructionHead(nn.Module):
    def __init__(self, conv_channels: int, output_dim: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv1d(conv_channels, output_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conv_channels: int,
        kernel_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool, # BiLSTM
    ):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(input_dim, conv_channels, kernel_size, dropout)
        self.encoder = Encoder(conv_channels, hidden_size, num_layers, dropout, bidirectional=bidirectional)
        self.decoder = Decoder(conv_channels, hidden_size, num_layers, dropout, bidirectional=bidirectional)
        self.reconstruction_head = ReconstructionHead(conv_channels, input_dim, kernel_size)

    def forward(self, x):
        # 特徵提取
        conv_features = self.feature_extractor(x)
        # Encoder 壓縮
        hidden, cell = self.encoder(conv_features)
        # Decoder 重建
        decoder_input = torch.zeros_like(conv_features)
        decoded_features = self.decoder(decoder_input, hidden, cell)
        # 重建
        reconstruction = self.reconstruction_head(decoded_features)
        return reconstruction


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for (batch_x, _) in loader:
        batch_x = batch_x.to(device)

        optimizer.zero_grad()
        reconstruction = model(batch_x)
        loss = criterion(reconstruction, batch_x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)


def evaluate_reconstruction_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (batch_x, _) in loader:
            batch_x = batch_x.to(device)
            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)