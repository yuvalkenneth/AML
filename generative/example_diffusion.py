import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

LR = 1e-3
T = 1000
EPOCHS = 3000
SAMPLE_SIZE = 2000
BATCH_SIZE = SAMPLE_SIZE


def g_t(t):
    return torch.exp(5 * (t - 1))


class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.ln1 = nn.Linear(3, 64)
        self.ln2 = nn.Linear(64, 64)
        self.leaky_relu = nn.LeakyReLU()
        self.ln3 = nn.Linear(64, 2)


if __name__ == '__main__':
    data = np.random.uniform(-1, 1, (SAMPLE_SIZE, 2))
    data = torch.tensor(data, dtype=torch.float32)
