import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 构建DAGMM模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class DAGMM(nn.Module):
    def __init__(self, input_dim, latent_dim, num_gmm):
        super(DAGMM, self).__init__()
        self.encoder = Autoencoder(input_dim, latent_dim)
        self.estimation = GaussianMixtureModel(latent_dim, num_gmm)

    def forward(self, x):
        x_recon, z = self.encoder(x)
        gamma = self.estimation(x_recon, z)
        return x_recon, z, gamma

class GaussianMixtureModel(nn.Module):
    def __init__(self, latent_dim, num_gmm):
        super(GaussianMixtureModel, self).__init__()
        self.num_gmm = num_gmm
        self.latent_dim = latent_dim
        self.phi = nn.Parameter(torch.ones(num_gmm) / num_gmm)
        self.mu = nn.Parameter(torch.randn(num_gmm, latent_dim))
        self.cov = nn.Parameter(torch.ones(num_gmm, latent_dim))

    def forward(self, x_recon, z):
        # 计算负对数似然
        diff = (z.unsqueeze(1) - self.mu.unsqueeze(0))
        cov_inverse = torch.exp(-self.cov).unsqueeze(0)
        exponent = -0.5 * torch.sum(diff.pow(2) * cov_inverse, dim=2)
        likelihood = torch.exp(exponent) / torch.sqrt((2 * np.pi) ** self.latent_dim * torch.prod(torch.exp(self.cov), dim=1))
        gamma_numerator = self.phi.unsqueeze(0) * likelihood
        gamma = gamma_numerator / torch.sum(gamma_numerator, dim=1, keepdim=True)
        return gamma