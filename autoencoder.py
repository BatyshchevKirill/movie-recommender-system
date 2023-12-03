import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)


class Autoencoder(nn.Module):
    """
    Autoencoder neural network for dimensionality reduction.

    Parameters:
    - input_dim (int): Dimensionality of the input data.
    - hidden_dim (int): Dimensionality of the hidden layer.
    - enc_dim (int): Dimensionality of the encoding layer.
    - l1_weight (float): Weight for L1 regularization.
    - l2_weight (float): Weight for L2 regularization.
    - device (torch.device): Device for computation (cuda or cpu).

    Methods:
    - forward(x): Forward pass through the autoencoder.
    - encode(x): Encodes input data into the encoding layer.
    - reg_loss(): Computes the regularization loss for L1 and L2 regularization.
    """
    def __init__(self, input_dim, hidden_dim, enc_dim, l1_weight, l2_weight, device):
        super(Autoencoder, self).__init__()

        # Encoder layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, enc_dim),
            nn.ReLU(),
        )

        # Decoder layer
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )

        self.l1 = l1_weight
        self.l2 = l2_weight
        self.device = device

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Parameters:
        - x (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Reconstructed output.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        """
        Encodes input data into the encoding layer.

        Parameters:
        - x (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Encoded representation.
        """
        return self.encoder(x)

    def reg_loss(self):
        """
        Computes the regularization loss for L1 and L2 regularization.

        Returns:
        - torch.Tensor: Regularization loss.
        """
        l1 = torch.tensor(0.0).to(self.device)
        l2 = torch.tensor(0.0).to(self.device)

        for param in self.parameters():
            l1 += torch.norm(param, 1)
            l2 += torch.norm(param, 2)

        return self.l1 * l1 + self.l2 * (l2**2)


def train_and_encode(
    loader: torch.utils.data.DataLoader,
    input_dim: int = 28,
    hidden_dim: int = 16,
    enc_dim: int = 8,
    epochs: int = 100,
    noise: float = 0.0,
    l1_weight: float = 0.0,
    l2_weight: float = 0.0,
    cpt_folder: str = "models/",
    data_folder: str = "data/interim/",
):
    """
    Trains an autoencoder on input data and saves the model and encoded data.

    Parameters:
    - loader (torch.utils.data.DataLoader): DataLoader for input data.
    - input_dim (int): Dimensionality of the input data.
    - hidden_dim (int): Dimensionality of the hidden layer.
    - enc_dim (int): Dimensionality of the encoding layer.
    - epochs (int): Number of training epochs.
    - noise (float): Scale of random noise added to input data.
    - l1_weight (float): Weight for L1 regularization.
    - l2_weight (float): Weight for L2 regularization.
    - cpt_folder (str): Folder to save the trained model.
    - data_folder (str): Folder to save the encoded data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(
        input_dim, hidden_dim, enc_dim, l1_weight, l2_weight, device
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    bar = tqdm(range(epochs))
    for epoch in bar:
        # Train loop
        total_loss = 0
        for batch in loader:
            batch = batch[0].to(device)
            # Add scaled random noise
            batch_noisy = batch + noise * torch.randn(*batch.shape, device=device)

            optim.zero_grad()

            out = model(batch_noisy)
            loss = criterion(out, batch)
            loss += model.reg_loss()

            total_loss += loss.item()

            loss.backward()
            optim.step()
        bar.set_postfix_str(
            f"Epoch {epoch + 1}. Average MSE loss + regularization: {total_loss / len(loader)}"
        )

    # Generate a name based on hyperparameters
    name = f"ae_{hidden_dim}_{enc_dim}_{epochs}_{noise}_{l1_weight:.4f}_{l2_weight:.4f}".replace(
        ".", "_"
    )
    torch.save(model, cpt_folder + name + ".pt")

    # Make predictions
    model.eval()
    encoded_data_list = []
    with torch.no_grad():
        for batch in loader:
            input_data = batch[0].to(device)
            encoded_data = model.encode(input_data)
            encoded_data_list.append(encoded_data.cpu().numpy())

    concatenated_data = np.concatenate(encoded_data_list, axis=0)

    # Save encoded data as a csv file
    column_names = [str(i) for i in range(concatenated_data.shape[1])]
    encoded_dataframe = pd.DataFrame(concatenated_data, columns=column_names)
    encoded_dataframe.to_csv(data_folder + "encoded_" + name + ".csv", index=False)
