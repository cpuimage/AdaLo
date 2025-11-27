"""
Convolutional Variational Auto-Encoder on MNIST – PyTorch implementation
"""
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

from adalo import AdaLo

optimizer = AdaLo


# --------------------------------------------------
# 1. Fix random seed
# --------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


seed_everything()

# --------------------------------------------------
# 2. Hyper-parameters
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 30
lr = 0.0
latent_dim = 2

# --------------------------------------------------
# 3. Data
# --------------------------------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

all_imgs = torch.cat([train_set.data, test_set.data]).unsqueeze(1).float() / 255.0
all_labels = torch.cat([train_set.targets, test_set.targets])

dataset = TensorDataset(all_imgs, all_labels)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# --------------------------------------------------
# 4. Model
# --------------------------------------------------
class Encoder(nn.Module):
    """Convolutional encoder outputting z_mean, z_log_var"""

    def __init__(self, latent_dim=latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 7x7
        self.fc1 = nn.Linear(7 * 7 * 64, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Convolutional decoder: input latent vector, output 28x28x1"""

    def __init__(self, latent_dim=latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 64)
        self.conv_t1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)  # 14x14
        self.conv_t2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 28x28
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(z.size(0), 64, 7, 7)
        z = F.relu(self.conv_t1(z))
        z = F.relu(self.conv_t2(z))
        z = self.conv_out(z)
        if not self.training:
            z = torch.sigmoid(z)
        return z


def reparameterize(mu, logvar):
    """Sample z = mu + eps * sigma"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------------------------------
# 5. Loss function
# --------------------------------------------------
def loss_fn(x, logits, mu, logvar):
    # Reconstruction loss
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction='sum') / x.size(0)
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return bce + kld, bce, kld


# --------------------------------------------------
# 6. Training
# --------------------------------------------------
encoder = Encoder().to(device)
decoder = Decoder().to(device)
log_dir = os.path.join("training_log", optimizer.__name__)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
summary_writer = tensorboard.SummaryWriter(log_dir)
print("Tensorboard: tensorboard --logdir={}".format(os.path.abspath(log_dir)))
opt = optimizer(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
if device.type == 'cuda':
    scaler = torch.GradScaler()
    if torch.cuda.is_bf16_supported() and torch.cuda.get_device_properties(0).major >= 8:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
else:
    scaler = None
    amp_dtype = torch.float32
    amp_context = nullcontext()

for epoch in range(1, epochs + 1):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    start_time = time.time()
    dataset_count = len(loader)
    global_step = epoch * dataset_count
    for x, _ in tqdm(loader, leave=False, desc=f"Epoch {epoch}"):
        x = x.to(device)

        loss_container = []


        def closure(data=x):
            opt.zero_grad()
            with amp_context:
                mu, logvar = encoder(data)
                z = reparameterize(mu, logvar)
                x_hat = decoder(z)
                _loss, _bce, _kld = loss_fn(data, x_hat, mu, logvar)
                if scaler is None:
                    _loss.backward()
            loss_container.extend([_loss.item(), _bce.item(), _kld.item()])
            return _loss


        # AdaLo requires a closure:
        #  1. closure must zero gradients, run forward & backward, and return the loss.
        #  2. If mixed-precision training is used, pass the GradScaler;
        #  the optimizer internally calls scaler.scale(loss).backward(), scaler.unscale_(self), scaler.step(self), and scaler.update().
        opt.step(closure, scaler)
        loss, bce, kld = loss_container

        total_loss += loss
        recon_loss += bce
        kl_loss += kld
        global_step += 1
        if summary_writer is not None:
            summary_writer.add_scalar("loss", loss, global_step)
            summary_writer.add_scalar("recon", recon_loss, global_step)
            summary_writer.add_scalar("kl", kl_loss, global_step)

    epoch_time = time.time() - start_time
    log_message = f"Epoch {epoch:02d} | Time: {epoch_time:.2f}s | loss={total_loss / dataset_count:.4f} " \
                  f"recon={recon_loss / dataset_count:.4f} kl={kl_loss / dataset_count:.4f}"
    print(log_message)

    with open(os.path.join(log_dir, f"{timestamp}_training_log.txt"), "a") as f:
        print(log_message, file=f)


# --------------------------------------------------
# 7. Visualization
# --------------------------------------------------
@torch.no_grad()
def plot_latent_space(n=30, figsize=15):
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    encoder.eval()
    decoder.eval()
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
            x_dec = decoder(z).squeeze().cpu().numpy()
            figure[i * digit_size:(i + 1) * digit_size,
            j * digit_size:(j + 1) * digit_size] = x_dec

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='Greys_r')
    plt.xticks(np.arange(0, n * digit_size, digit_size), np.round(grid_x, 2))
    plt.yticks(np.arange(0, n * digit_size, digit_size), np.round(grid_y, 2))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("2D latent space manifold")
    plt.savefig(os.path.join(log_dir, f"{timestamp}_latent_space_epoch_{epoch}.png"), dpi=120)
    plt.close()


@torch.no_grad()
def plot_label_clusters(data_loader, max_samples=5000):
    zs, ys = [], []
    encoder.eval()
    for x, y in data_loader:
        x = x.to(device)
        mu, _ = encoder(x)
        zs.append(mu.cpu())
        ys.append(y)
        if len(zs) * batch_size >= max_samples:
            break
    zs = torch.cat(zs).numpy()
    ys = torch.cat(ys).numpy()
    plt.figure(figsize=(12, 10))
    plt.scatter(zs[:, 0], zs[:, 1], c=ys, cmap='tab10')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Latent space clustering by digit class")
    plt.savefig(os.path.join(log_dir, f"{timestamp}_latent_clustering_epoch_{epoch}.png"), dpi=120)
    plt.close()


# Run visualization
plot_latent_space()
plot_label_clusters(loader)
