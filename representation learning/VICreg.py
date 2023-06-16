import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import plotly.express as px
import augmentations
from models import Encoder, Projector

WEIGHT_DECAY = 10 ** -6

BETAS = (0.9, 0.999)

GAMMA = 1
EPSILON = 10 ** -4
LAMBDA = 25
MU = 25
NU = 1
PROJ_DIM = 512
ENCODER_DIM = 128
BATCH_SIZE = 256
LR = 3 * (10 ** -4)
EPOCHS = 40


def invariance_term(x, y):
    return F.mse_loss(x, y, reduction='mean')


def variance_term(x):
    sigma_j = torch.var(x, dim=0)
    return F.hinge_embedding_loss(sigma_j, target=torch.zeros_like(x) - 1, margin=GAMMA, reduction='mean')


def covariance_term(x):
    batch_size = x.shape[0]
    dim = x.shape[1]
    x_hat = torch.mean(x, dim=0)
    y = (x - x_hat).view(batch_size, dim, 1)
    c_z = torch.sum(torch.matmul(y, y.transpose(1, 2)), dim=0) / (batch_size - 1)
    c_z.diagonal().zero_()
    return torch.sum(torch.square(c_z)) / dim


def vic_train(encoder, projector, data, optimizer):
    encoder.train()
    projector.train()
    for epoch in range(EPOCHS):
        for images, labels in data:
            optimizer.zero_grad()
            aug1 = torch.stack([augmentations.train_transform(image) for image in images])
            aug2 = torch.stack([augmentations.train_transform(image) for image in images])
            z = projector(encoder.encode(aug1))
            z_hat = projector(encoder.encode(aug2))
            invariance = invariance_term(z, z_hat)
            v_z = variance_term(z)
            v_z_hat = variance_term(z_hat)
            cov_z = covariance_term(z)
            cov_z_hat = covariance_term(z_hat)
            loss = LAMBDA * invariance + MU * (v_z + v_z_hat) + NU * (cov_z + cov_z_hat)
            loss.backward()
            optimizer.step()
            invariance_loss.append(invariance.item())
            variance_loss.append((v_z + v_z_hat).item())
            covariance_loss.append((cov_z + cov_z_hat).item())
    return invariance_loss, variance_loss, covariance_loss


def test_encoder(encoder, data):
    encoder.eval()
    for epoch in range(EPOCHS):
        for images, labels in data:
            aug1 = torch.stack([augmentations.test_transform(image) for image in images])
            aug2 = torch.stack([augmentations.test_transform(image) for image in images])
            z = encoder.encode(aug1)
            z_hat = encoder.encode(aug2)
            invariance = invariance_term(z, z_hat)
            v_z = variance_term(z)
            v_z_hat = variance_term(z_hat)
            cov_z = covariance_term(z)
            cov_z_hat = covariance_term(z_hat)
            loss = LAMBDA * invariance + MU * (v_z + v_z_hat) + NU * (cov_z + cov_z_hat)
        invariance_loss.append(invariance.item())
        variance_loss.append((v_z + v_z_hat).item())
        covariance_loss.append((cov_z + cov_z_hat).item())
    return invariance_loss, variance_loss, covariance_loss


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = Encoder(D=ENCODER_DIM, device=device)
    proj = Projector(D=enc.fc[2].out_features, proj_dim=PROJ_DIM)
    optim = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=LR, betas=BETAS,
                             weight_decay=WEIGHT_DECAY)
    invariance_loss, variance_loss, covariance_loss = vic_train(enc, proj, trainloader, optim)
    test_invariance_loss, test_variance_loss, test_covariance_loss = test_encoder(enc, testloader)
    fig1 = px.line(x=range(len(invariance_loss)), y=invariance_loss, title='Invariance Loss')
    fig1.add_scatter(x=range(EPOCHS), y=test_invariance_loss, name='Test')
    fig1.show()
    fig2 = px.line(x=range(len(variance_loss)), y=variance_loss, title='Variance Loss')
    fig2.add_scatter(x=range(EPOCHS), y=test_variance_loss, name='Test')
    fig2.show()
    fig3 = px.line(x=range(len(covariance_loss)), y=covariance_loss, title='Covariance Loss')
    fig3.add_scatter(x=range(EPOCHS), y=test_covariance_loss, name='Test')
    fig3.show()



