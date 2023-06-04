import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

LR = 1e-3
T = 1000
EPOCHS = 3000
SAMPLE_SIZE = 3000
BATCH_SIZE = SAMPLE_SIZE


def sigma_t(t):
    return torch.exp(5 * (t - 1)).requires_grad_(True)


class TrainSet(Dataset):
    def __init__(self, min_value, max_value, dim, size, conditional=None):
        self.size = size
        self.dim = dim
        self.min = min_value
        self.max = max_value
        points = np.random.uniform(min_value, max_value, (size, dim))
        self.points = torch.tensor(points, dtype=torch.float32)
        if conditional is not None:
            self.classes = np.apply_along_axis(conditional, 1, points)
        self.conditional = conditional

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.conditional is not None:
            return self.points[idx], self.classes[idx]
        return self.points[idx]

    def get_dimension(self):
        return self.dim


class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.ln1 = nn.Linear(3, 64)
        self.ln2 = nn.Linear(64, 64)
        self.leaky_relu = nn.LeakyReLU()
        self.ln3 = nn.Linear(64, 2)

    def forward(self, x, t):
        x = torch.cat((x, t), dim=1)
        x = self.ln1(x)
        x = self.leaky_relu(x)
        x = self.ln2(x)
        x = self.leaky_relu(x)
        x = self.ln3(x)
        return x


class ConditionedDenoiser(nn.Module):
    def __init__(self, num_classes):
        super(ConditionedDenoiser, self).__init__()
        self.embedding = nn.Embedding(num_classes, 10)
        self.ln1 = nn.Linear(13, 128)
        self.ln2 = nn.Linear(128, 128)
        self.leaky_relu = nn.LeakyReLU()
        self.ln3 = nn.Linear(128, 64)
        self.ln4 = nn.Linear(64, 2)

    def forward(self, x, t, c):
        c = self.embedding(c)
        x = torch.cat((x, t, c), dim=1)
        x = self.ln1(x)
        x = self.leaky_relu(x)
        x = self.ln2(x)
        x = self.leaky_relu(x)
        x = self.ln3(x)
        x = self.leaky_relu(x)
        x = self.ln4(x)
        return x


def train_diffusion(diffusion_model, points, epochs, batch_size, dimension, noise_scheduler):
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=LR)
    loss_over_batches = []
    for epoch in tqdm(range(epochs)):
        for batch in points:
            optimizer.zero_grad()
            epsilon = torch.randn(batch_size, dimension)
            t = torch.FloatTensor(batch_size).uniform_().reshape(batch_size, 1)
            x_t = batch + (noise_scheduler(t) * epsilon)
            expected_noise = diffusion_model(x_t, t)
            loss = F.mse_loss(expected_noise, epsilon)
            loss.backward()
            optimizer.step()
            loss_over_batches.append(loss.item())
    return loss_over_batches


def train_conditional_diffusion(conditioned_diffusion_model, points, epochs, batch_size, dimension,
                                noise_scheduler):
    optimizer = torch.optim.Adam(conditioned_diffusion_model.parameters(), lr=LR)
    loss_over_batches = []

    for epoch in tqdm(range(epochs)):
        for batch in points:
            optimizer.zero_grad()
            epsilon = torch.randn(batch_size, dimension)
            coordinates = batch[0]
            classes = batch[1].long()
            t = torch.FloatTensor(batch_size).uniform_().reshape(batch_size, 1)
            x_t = coordinates + (noise_scheduler(t)) * epsilon
            expected_noise = conditioned_diffusion_model(x_t, t, classes)
            loss = F.mse_loss(expected_noise, epsilon)
            loss.backward()
            optimizer.step()
            loss_over_batches.append(loss.item())
    return loss_over_batches


def point_sampling(denoiser, step_size, noise_scheduler, dim, num_samples=1, seed=None, classes=None,
                   noise=None, stochastic=False):
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(num_samples, dim, dtype=denoiser.ln1.weight.dtype) if noise is None else noise.detach(
    ).clone()
    trajectory = [z.detach().tolist()[0]]
    if step_size != 0:
        for t in np.arange(1, 0, step_size):
            t = torch.tensor(t, requires_grad=True, dtype=denoiser.ln1.weight.dtype).reshape(1, 1)
            t.retain_grad()
            sigma = noise_scheduler(t)
            sigma_derivative = torch.autograd.grad(sigma, t)[0]
            if classes is None:
                estimated_noise = denoiser(z, t.expand(num_samples, 1))
            else:
                estimated_noise = denoiser(z, t.expand(num_samples, 1), classes.expand(num_samples))
            z_hat = z - sigma * estimated_noise
            if stochastic:
                z_hat += sigma * torch.randn_like(z_hat) * 0.2
            score = (z_hat - z) / (sigma ** 2)
            dz = -sigma_derivative * sigma * score * step_size
            z = z + dz
            trajectory.append(z.detach().tolist()[0])
    return z, trajectory


def custom_schedule_sampling(denoiser, steps, noise_scheduler, dim, num_samples=1, classes=None,
                             noise=None):
    z = torch.randn(num_samples, dim,
                    dtype=denoiser.ln1.weight.dtype) if noise is None else noise.detach().clone()
    trajectory = [z.detach().tolist()[0]]
    for ind, t in enumerate(steps):
        t = torch.tensor(t, requires_grad=True, dtype=denoiser.ln1.weight.dtype).reshape(1, 1)
        t.retain_grad()
        sigma = noise_scheduler(t)
        sigma_derivative = torch.autograd.grad(sigma, t)[0]
        if classes is None:
            estimated_noise = denoiser(z, t.expand(num_samples, 1))
        else:
            estimated_noise = denoiser(z, t.expand(num_samples, 1), classes.expand(num_samples))
        z_hat = z - sigma * estimated_noise
        score = (z_hat - z) / (sigma ** 2)
        dz = -sigma_derivative * sigma * score * (steps[ind] - (ind if ind == 0 else steps[ind - 1]))
        z = z + dz
        trajectory.append(z.detach().tolist()[0])
    return z, trajectory


def snr(noise_scheduler, factor, x):
    return (factor(x) ** 2) / (noise_scheduler(x) ** 2)


def point_estimation(point, noise_scheduler, time_step, denoiser, label=None):
    with torch.no_grad():
        general_loss = []
        for i in range(int(1 / time_step)):
            t = torch.FloatTensor(1).uniform_().reshape(1, 1)
            noise = torch.randn(1, 2)
            x_t = point + (noise_scheduler(t) * noise)
            if label is None:
                x_0 = x_t - (sigma_t(t) * denoiser(x_t, torch.tensor(t)))
            else:
                x_0 = x_t - (denoiser(x_t, t, label) * noise_scheduler(t))
            SNR = (snr(noise_scheduler, lambda x: 1, (t - time_step)) - snr(noise_scheduler, lambda x: 1, t))
            loss = F.mse_loss(x_0.squeeze(), point.squeeze())
            general_loss.append((loss * SNR).detach().numpy())

        return -np.mean(general_loss) * (1 / (time_step * 2))
