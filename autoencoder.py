import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, enc_dim, l1_weight, l2_weight, device):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, enc_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

        self.l1 = l1_weight
        self.l2 = l2_weight
        self.device = device

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def reg_loss(self):
        l1 = torch.tensor(0.0).to(self.device)
        l2 = torch.tensor(0.0).to(self.device)

        for param in self.parameters():
            l1 += torch.norm(param, 1)
            l2 += torch.norm(param, 2)

        return self.l1 * l1 + self.l2 * (l2 ** 2)


def train_and_encode(loader: torch.utils.data.DataLoader,
                     input_dim: int = 28,
                     hidden_dim: int = 16,
                     enc_dim: int = 8,
                     epochs: int = 100,
                     noise: float = 0.0,
                     l1_weight: float = 0.0,
                     l2_weight: float = 0.0,
                     cpt_folder: str = "models/",
                     data_folder: str = "data/interim"
                     ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim, hidden_dim, enc_dim, l1_weight, l2_weight, device).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    bar = tqdm(range(epochs))
    for epoch in bar:
        total_loss = 0
        for batch in loader:
            batch = batch[0].to(device)
            batch_noisy = batch + noise * torch.randn(*batch.shape, device=device)

            optim.zero_grad()

            out = model(batch_noisy)
            loss = criterion(out, batch)
            loss += model.reg_loss()

            total_loss += loss.item()

            loss.backward()
            optim.step()
        bar.set_postfix_str(f"Epoch {epoch + 1}. Average MSE loss + regularization: {total_loss / len(loader)}")

    name = f"ae_{hidden_dim}_{enc_dim}_{epochs}_{noise}_{l1_weight:.4f}_{l2_weight:.4f}".replace(".", "_")
    torch.save(model, cpt_folder + name + '.pt')


