# import torch
# import torch.nn.functional as F
# import torchvision
# from torch.utils.data import Dataset
# from torchvision import transforms
# from tqdm import tqdm
#
# import augmentations
# from models import Encoder, Projector
#
# WEIGHT_DECAY = 10 ** -6
#
# BETAS = (0.9, 0.999)
#
# GAMMA = 1
# EPSILON = 10 ** -4
# LAMBDA = 25
# MU = 25
# NU = 1
# PROJ_DIM = 512
# ENCODER_DIM = 128
# BATCH_SIZE = 256
# LR = 3 * (10 ** -4)
# EPOCHS = 40
#
#
# def collate_fn(batch):
#     images, labels = zip(*batch)
#     aug1 = torch.stack([augmentations.train_transform(image) for image in images])
#     aug2 = torch.stack([augmentations.train_transform(image) for image in images])
#     return aug1, aug2
#
#
# # def test_collate_fn(batch):
# #     images, labels = zip(*batch)
# #     return torch.stack([augmentations.test_transform(image) for image in images]), torch.tensor(labels)
#
#
# def invariance_term(x, y):
#     losses = F.mse_loss(x, y, reduction='none')
#
#     return losses.mean()
#
#
# def variance_term(x):
#     sigma_j = (torch.var(x, dim=0) + EPSILON) ** 0.5
#     return F.hinge_embedding_loss(sigma_j, target=torch.full_like(sigma_j, -1),
#                                   margin=GAMMA, reduction='mean')
#
#
# def covariance_term(x):
#     batch_size = x.shape[0]
#     dim = x.shape[1]
#     x_hat = torch.mean(x, dim=0)
#     y = x - x_hat
#     m = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))
#     c_z = torch.sum(m, dim=0) / batch_size - 1
#     c_z.diagonal().zero_()
#     return torch.sum(c_z ** 2) / dim
#
#
# def vic_train(encoder, projector, data, optimizer, dev='cuda'):
#     encoder.train()
#     projector.train()
#     invariance_loss, variance_loss, covariance_loss = [], [], []
#     for epoch in tqdm(range(EPOCHS)):
#         for aug1, aug2 in data:
#             optimizer.zero_grad()
#             z = projector(encoder.encode(aug1.to(dev)))
#             z_hat = projector(encoder.encode(aug2.to(dev)))
#             invariance = invariance_term(z, z_hat)
#             v_z = variance_term(z)
#             v_z_hat = variance_term(z_hat)
#             cov_z = covariance_term(z)
#             cov_z_hat = covariance_term(z_hat)
#             loss = LAMBDA * invariance + MU * (v_z + v_z_hat) + NU * (cov_z + cov_z_hat)
#             loss.backward()
#             optimizer.step()
#             invariance_loss.append(invariance.item())
#             variance_loss.append((v_z + v_z_hat).item())
#             covariance_loss.append((cov_z + cov_z_hat).item())
#     return invariance_loss, variance_loss, covariance_loss
#
#
# def test_encoder(encoder, data):
#     encoder.eval()
#     for epoch in range(EPOCHS):
#         for images, labels in data:
#             aug1 = torch.stack([augmentations.test_transform(image) for image in images])
#             aug2 = torch.stack([augmentations.test_transform(image) for image in images])
#             z = encoder.encode(aug1)
#             z_hat = encoder.encode(aug2)
#             invariance = invariance_term(z, z_hat)
#             v_z = variance_term(z)
#             v_z_hat = variance_term(z_hat)
#             cov_z = covariance_term(z)
#             cov_z_hat = covariance_term(z_hat)
#             loss = LAMBDA * invariance + MU * (v_z + v_z_hat) + NU * (cov_z + cov_z_hat)
#         invariance_loss.append(invariance.item())
#         variance_loss.append((v_z + v_z_hat).item())
#         covariance_loss.append((cov_z + cov_z_hat).item())
#     return invariance_loss, variance_loss, covariance_loss
#
#
# if __name__ == '__main__':
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
#                                             transform=transforms.ToTensor())
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
#                                               collate_fn=collate_fn)
#     # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
#     #                                        transform=transforms.ToTensor())
#     # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
#     #                                          collate_fn=test_collate_fn)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     enc = Encoder(D=ENCODER_DIM, device=device).to(device)
#     proj = Projector(D=enc.fc[2].out_features, proj_dim=PROJ_DIM).to(device)
#     optim = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=LR, betas=BETAS,
#                              weight_decay=WEIGHT_DECAY)
#     invariance_loss, variance_loss, covariance_loss = vic_train(enc, proj, trainloader, optim, device)
#     # test_invariance_loss, test_variance_loss, test_covariance_loss = test_encoder(enc, testloader)
#     # fig1 = px.line(x=range(len(invariance_loss)), y=invariance_loss, title='Invariance Loss')
#     # fig1.add_scatter(x=range(EPOCHS), y=test_invariance_loss, name='Test')
#     # fig1.show()
#     # fig2 = px.line(x=range(len(variance_loss)), y=variance_loss, title='Variance Loss')
#     # fig2.add_scatter(x=range(EPOCHS), y=test_variance_loss, name='Test')
#     # fig2.show()
#     # fig3 = px.line(x=range(len(covariance_loss)), y=covariance_loss, title='Covariance Loss')
#     # fig3.add_scatter(x=range(EPOCHS), y=test_covariance_loss, name='Test')
# #     fig3.show()
import os

import plotly.express as px
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
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
EPOCHS = 25


def collate_fn(batch):
    images, labels = zip(*batch)
    aug1 = torch.stack([augmentations.train_transform(image) for image in images])
    aug2 = torch.stack([augmentations.train_transform(image) for image in images])
    return aug1, aug2


def invariance_term(x, y):
    losses = F.mse_loss(x, y, reduction='none')
    return losses.mean()


def variance_term(x):
    sigma_j = (torch.var(x, dim=0) + EPSILON) ** 0.5
    return F.hinge_embedding_loss(sigma_j, target=torch.full_like(sigma_j, -1),
                                  margin=GAMMA, reduction='mean')


def covariance_term(x):
    batch_size = x.shape[0]
    dim = x.shape[1]
    x_hat = torch.mean(x, dim=0)
    y = x - x_hat
    m = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))
    c_z = torch.sum(m, dim=0) / batch_size - 1
    c_z.diagonal().zero_()
    return torch.sum(c_z ** 2) / dim


def vic_train(encoder, projector, data, optimizer, dev='cuda'):
    encoder.train()
    projector.train()
    invariance_loss, variance_loss, covariance_loss = [], [], []
    for epoch in tqdm(range(EPOCHS)):
        for aug1, aug2 in data:
            optimizer.zero_grad()
            z = projector(encoder.encode(aug1.to(dev)))
            z_hat = projector(encoder.encode(aug2.to(dev)))
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
    test_invariance, test_variance, test_covariance = [], [], []
    with torch.no_grad():
        for epoch in range(EPOCHS):
            batch_invariance_loss, batch_variance_loss, batch_covariance_loss = [], [], []
            for aug1, aug2 in data:
                z = encoder.encode(aug1)
                z_hat = encoder.encode(aug2)
                invariance = invariance_term(z, z_hat)
                v_z = variance_term(z)
                v_z_hat = variance_term(z_hat)
                cov_z = covariance_term(z)
                cov_z_hat = covariance_term(z_hat)
                loss = LAMBDA * invariance + MU * (v_z + v_z_hat) + NU * (cov_z + cov_z_hat)
                batch_invariance_loss.append(invariance.item())
                batch_variance_loss.append((v_z + v_z_hat).item())
                batch_covariance_loss.append((cov_z + cov_z_hat).item())

            test_invariance.append(np.mean(batch_invariance_loss))
            test_variance.append(np.mean(batch_variance_loss))
            test_covariance.append(np.mean(batch_covariance_loss))
        return test_invariance, test_variance, test_covariance


if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                              collate_fn=collate_fn)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                             collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = Encoder(D=ENCODER_DIM, device=device).to(device)
    proj = Projector(D=enc.fc[2].out_features, proj_dim=PROJ_DIM).to(device)
    optim = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=LR, betas=BETAS,
                             weight_decay=WEIGHT_DECAY)
    if os.path.exists('encoder_weights.pth') and os.path.exists('projector_weights.pth'):
        enc.load_state_dict(torch.load('encoder_weights.pth', map_location="cpu"))
        proj.load_state_dict(torch.load('projector_weights.pth', map_location="cpu"))
    else:
        invariance_loss, variance_loss, covariance_loss = vic_train(enc, proj, trainloader, optim, device)
        torch.save(enc.state_dict(), 'encoder.pth')
        torch.save(proj.state_dict(), 'projector.pth')

    test_invariance_loss, test_variance_loss, test_covariance_loss = test_encoder(enc, testloader)
    rn = np.array(range(EPOCHS)) * BATCH_SIZE
    fig1 = px.line(x=range(len(invariance_loss)), y=invariance_loss, title='Invariance Loss')
    fig1.add_scatter(x=rn, y=test_invariance_loss, name='Test')
    fig1.show()
    fig2 = px.line(x=range(len(variance_loss)), y=variance_loss, title='Variance Loss')
    fig2.add_scatter(x=rn, y=test_variance_loss, name='Test')
    fig2.show()
    fig3 = px.line(x=rn, y=covariance_loss, title='Covariance Loss')
    fig3.add_scatter(x=rn, y=test_covariance_loss, name='Test')
    fig3.show()
