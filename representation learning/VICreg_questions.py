import os
import random

import faiss
import numpy as np
import plotly.express as px
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

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
EPOCHS = 10
NEIGHBORS_EPOCH = 1


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        return (self.transform(image), self.transform(image)), self.labels[index]

    def __len__(self):
        return len(self.images)


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
    c_z = torch.sum(m, dim=0) / (batch_size - 1)
    c_z.diagonal().zero_()
    return torch.sum(c_z ** 2) / dim


class TestingDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        return self.transform(image), self.labels[index]

    def __len__(self):
        return len(self.images)


class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class NearestNeighborsImages(Dataset):
    def __init__(self, data, neighbors):
        self.data = data
        self.neighbors = neighbors

    def __getitem__(self, index):
        image = self.data[index]
        neighbor = np.random.choice(self.neighbors[index])
        return image, self.data[neighbor]

    def __len__(self):
        return len(self.data)


def pca_tsne(enc, data):
    pca = PCA()
    tsne = TSNE()
    for image, target in data:
        h = enc.encode(image.to(device))
        tsne_data = tsne.fit_transform(h.cpu())
        pca_data = pca.fit_transform(h.cpu())

        tsne_plot = sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=target, palette="Paired")
        tsne_plot.set(title="TSNE")
        tsne_plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        pca_plot = sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=target, palette="Paired")
        pca_plot.set(title="PCA")
        pca_plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


def linear_probing(encoder_dim, num_classes, train_loader, test_loader, enc, linear_probe, dev, path_name):
    enc.eval()
    for par in enc.parameters():
        par.requires_grad = False
    optimizer = torch.optim.Adam(linear_probe.parameters())
    criterion = nn.CrossEntropyLoss()
    test_acc = 0
    if os.path.exists(path_name):
        linear_probe.load_state_dict(torch.load(path_name, map_location="cuda:0" if
        torch.cuda.is_available() else "cpu"))
    else:
        linear_probe.train()
        for epoch in range(EPOCHS):
            for images, labels in train_loader:
                optimizer.zero_grad()
                h = enc.encode(images.to(dev))
                logits = linear_probe(h)
                loss = criterion(logits, labels.to(dev))
                loss.backward()
                optimizer.step()
        torch.save(linear_probe.state_dict(), path_name)
    return test_linear_probe(dev, enc, linear_probe, test_acc, test_loader)


def test_linear_probe(dev, enc, linear_probe, test_acc, test_loader):
    linear_probe.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            h = enc.encode(images.to(dev))
            logits = linear_probe(h)
            pred = torch.argmax(logits, dim=1)
            test_acc += torch.sum(pred == labels.to(dev)).item()
    return test_acc / len(test_loader.dataset)


def get_encoding(encoder, data_loader):
    encodings, targets = [], []
    for images, labels in data_loader:
        encodings.extend(encoder.encode(images.to(device)))
        targets.extend(labels)
    return torch.stack(encodings), torch.stack(targets)


def get_nearest_neighbors(data, k, enc):
    index = faiss.IndexFlatL2(ENCODER_DIM)
    codings = []
    dataset = []
    for images, labels in data:
        embeddings = enc.encode(images.to(device))
        index.add(embeddings.cpu())
        codings.append(embeddings)
        dataset.append(images)
    all_embeddings = torch.cat(codings, dim=0)
    data = torch.cat(dataset, dim=0)
    _, neighbors = index.search(all_embeddings.cpu(), k=k + 1)
    return NearestNeighborsImages(data, neighbors[:, 1:])


def vic_train(enc, proj, data, optimizer, dev, test_data):
    invariance_loss, variance_loss, covariance_loss = [], [], []
    test_invariance, test_variance, test_covariance = [], [], []
    rn = []
    counter = 0
    for epoch in tqdm(range(EPOCHS)):
        enc.train()
        proj.train()
        for images, labels in data:
            aug1, aug2 = images
            optimizer.zero_grad()
            z = proj(enc.encode(aug1.to(dev)))
            z_hat = proj(enc.encode(aug2.to(dev)))
            invariance = invariance_term(z, z_hat)
            v_z = variance_term(z)
            v_z_hat = variance_term(z_hat)
            cov_z = covariance_term(z)
            cov_z_hat = covariance_term(z_hat)
            invariance_loss.append(invariance.item())
            variance_loss.append((v_z + v_z_hat).item())
            covariance_loss.append((cov_z + cov_z_hat).item())
            loss = LAMBDA * invariance + NU * (cov_z + cov_z_hat) + MU * (v_z + v_z_hat)
            loss.backward()
            optimizer.step()
            counter += 1
        rn.append(counter)
        test_in, test_var, test_cov = test_encoder(enc, test_data)
        test_invariance.append(np.mean(test_in))
        test_variance.append(np.mean(test_var))
        test_covariance.append(np.mean(test_cov))

    plot_graphs(invariance_loss, variance_loss, covariance_loss, test_invariance, test_variance,
                test_covariance, rn)
    return enc


def plot_graphs(invariance_loss, variance_loss, covariance_loss, test_invariance_loss, test_variance_loss,
                test_covariance_loss, rn):
    fig1 = px.line(x=range(len(invariance_loss)), y=invariance_loss, title='Invariance Loss')
    fig1.add_scatter(x=rn, y=test_invariance_loss, name='Test')
    fig1.show()
    fig2 = px.line(x=range(len(variance_loss)), y=variance_loss, title='Variance Loss')
    fig2.add_scatter(x=rn, y=test_variance_loss, name='Test')
    fig2.show()
    fig3 = px.line(x=range(len(covariance_loss)), y=covariance_loss, title='Covariance Loss')
    fig3.add_scatter(x=rn, y=test_covariance_loss, name='Test')
    fig3.show()


def test_encoder(enc, data):
    enc.eval()
    test_invariance, test_variance, test_covariance = [], [], []
    with torch.no_grad():
        for images, labels in data:
            aug1, aug2 = images
            z = enc.encode(aug1.to(device))
            z_hat = enc.encode(aug2.to(device))
            invariance = invariance_term(z, z_hat)
            v_z = variance_term(z)
            v_z_hat = variance_term(z_hat)
            cov_z = covariance_term(z)
            cov_z_hat = covariance_term(z_hat)
            test_invariance.append(invariance.item())
            test_variance.append((v_z + v_z_hat).item())
            test_covariance.append((cov_z + cov_z_hat).item())
    return test_invariance, test_variance, test_covariance


def train_no_generated_neighbors(enc, proj, neighbors_loader, optimizer, dev):
    for epoch in range(NEIGHBORS_EPOCH):
        for image, neighbor in neighbors_loader:
            optimizer.zero_grad()
            z = proj(enc.encode(image.to(dev)))
            z_neighbor = proj(enc.encode(neighbor.to(dev)))
            invariance = invariance_term(z, z_neighbor)
            v_z = variance_term(z)
            v_z_neighbor = variance_term(z_neighbor)
            cov_z = covariance_term(z)
            cov_z_neighbor = covariance_term(z_neighbor)
            loss = LAMBDA * invariance + MU * (v_z + v_z_neighbor) + NU * (cov_z + cov_z_neighbor)
            loss.backward()
            optimizer.step()

    return enc


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def get_neighbors(enc, data, indices, k=10):
    embeddings = get_encoding(enc, data)[0]
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    k_nearest = torch.topk(dist_matrix, k=k + 1, dim=1, largest=False)[1]
    k_farthest = torch.topk(dist_matrix, k=k, dim=1, largest=True)[1]

    return k_nearest[indices, :], k_farthest[indices, :]


def retrieval(data, encoder, k=10):
    classes_indices = [-1] * 10
    classes_added = 0
    while classes_added < 10:
        index = random.randint(0, len(data))
        if classes_indices[data.dataset[index][1]] == -1:
            classes_indices[data.dataset[index][1]] = index
            classes_added += 1
    nearest, farthest = get_neighbors(encoder, data, classes_indices, k=k)
    plot_neighbors(data.dataset, nearest, farthest, k=k)


def plot_neighbors(data, nearest, farthest, k=10):
    fig, axs = plt.subplots(10, 2 * k + 1, figsize=(20, 20))
    for i in range(10):
        for j in range(k + 1):
            axs[i, j].imshow(data[nearest[i][j]][0].permute(1, 2, 0), interpolation='nearest')
            axs[i, j].axis('off')
            if j == 0:
                axs[i, j].set_title('source image')
            else:
                axs[i, j].set_title('Nearest {}'.format(j))
        for j in range(k):
            axs[i, k + j + 1].imshow(data[farthest[i][j]][0].permute(1, 2, 0), interpolation='nearest')
            axs[i, k + j + 1].axis('off')
            axs[i, k + j + 1].set_title('Farthest {}'.format(j + 1))
    fig.suptitle('Nearest and Farthest', fontsize=24)
    plt.show()


if __name__ == '__main__':
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transforms.ToTensor())
    train_image_set = ImageDataset(trainset.data, trainset.targets, transform=augmentations.train_transform)
    trainloader = torch.utils.data.DataLoader(train_image_set, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transforms.ToTensor())
    test_image_set = ImageDataset(testset.data, testset.targets, transform=augmentations.train_transform)
    testloader = torch.utils.data.DataLoader(test_image_set, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    projector = Projector(D=ENCODER_DIM, proj_dim=PROJ_DIM).to(device)
    optim = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=LR, betas=BETAS,
                             weight_decay=WEIGHT_DECAY)
    if os.path.exists('encoder.pth') and os.path.exists('projector.pth'):
        encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
        projector.load_state_dict(torch.load('projector.pth', map_location=device))
    else:
        encoder = vic_train(encoder, projector, trainloader, optim, device, testloader)
        torch.save(encoder.state_dict(), 'encoder.pth')
        torch.save(projector.state_dict(), 'projector.pth')

    freeze_model(encoder)
    trainset_downstream = TestingDataset(trainset.data, trainset.targets,
                                         transform=normalize)
    trainloader_downstream = torch.utils.data.DataLoader(trainset_downstream, batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         num_workers=4)
    testset_downstream = TestingDataset(testset.data, testset.targets, transform=augmentations.test_transform)
    testloader_downstream = torch.utils.data.DataLoader(testset_downstream, batch_size=BATCH_SIZE,
                                                        shuffle=False,
                                                        num_workers=4)

    whole_testset = torch.utils.data.DataLoader(testset_downstream, batch_size=len(testset.data),
                                                shuffle=False, num_workers=4)

    trainset_loader_no_aug = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                         shuffle=False, num_workers=4)

    # Q2 - PCA vs. T-SNE Visualizations
    pca_tsne(encoder, whole_testset)
    # Q3 - Linear Probing
    linear_probe = LinearProbe(ENCODER_DIM, 10).to(device)
    probing_acc = linear_probing(encoder_dim=ENCODER_DIM, num_classes=10, train_loader=trainloader_downstream,
                                 test_loader=testloader_downstream, enc=encoder, linear_probe=linear_probe,
                                 dev=device, path_name="probe_weights.pth")
    print('Linear Probing Accuracy: ', probing_acc)

    # Q4 - No variance term
    no_variance_encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    no_variance_projector = Projector(D=ENCODER_DIM, proj_dim=PROJ_DIM).to(device)
    no_variance_linear_probe = LinearProbe(ENCODER_DIM, 10).to(device)
    no_variance_encoder.load_state_dict(torch.load('encoder_no_variance.pth', map_location=device))
    no_variance_projector.load_state_dict(torch.load('projector_no_variance.pth', map_location=device))
    freeze_model(no_variance_encoder)
    no_variance_probe_acc = linear_probing(encoder_dim=ENCODER_DIM, num_classes=10,
                                           train_loader=trainloader_downstream,
                                           test_loader=testloader_downstream, enc=no_variance_encoder,
                                           linear_probe=no_variance_linear_probe, dev=device,
                                           path_name="probe_weights_no_variance.pth")
    print('No Variance Linear Probing Accuracy: ', no_variance_probe_acc)
    pca_tsne(no_variance_encoder, whole_testset)

    # Q6 - No Generated Neighbors
    nearest_neighbors_set = get_nearest_neighbors(trainset_loader_no_aug, 3, encoder)
    nearest_neighbor_loader = torch.utils.data.DataLoader(nearest_neighbors_set, batch_size=BATCH_SIZE,
                                                          shuffle=True,
                                                          num_workers=4)
    no_generated_encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    no_generated_projector = Projector(D=ENCODER_DIM, proj_dim=PROJ_DIM).to(device)
    if os.path.exists('no_generated_encoder.pth'):
        no_generated_encoder.load_state_dict(torch.load('no_generated_encoder.pth', map_location=device))
    else:
        no_generated_optim = torch.optim.Adam(
            list(no_generated_encoder.parameters()) + list(no_generated_projector.parameters()), lr=LR,
            betas=BETAS, weight_decay=WEIGHT_DECAY)
        train_no_generated_neighbors(no_generated_encoder, no_generated_projector, nearest_neighbor_loader,
                                     no_generated_optim, device)
        torch.save(no_generated_encoder.state_dict(), 'no_generated_encoder.pth')

    freeze_model(no_generated_encoder)
    pca_tsne(no_generated_encoder, whole_testset)
    no_generated_probe = LinearProbe(ENCODER_DIM, 10).to(device)
    probing_acc_no_neighbors = linear_probing(encoder_dim=ENCODER_DIM, num_classes=10,
                                              train_loader=trainloader_downstream,
                                              test_loader=testloader_downstream, enc=no_generated_encoder,
                                              linear_probe=no_generated_probe, dev=device,
                                              path_name="no_generated_probe_weights.pth")
    print('Linear Probing Accuracy: ', probing_acc_no_neighbors)

    # Q8 - Retrieval Evaluation
    retrieval(trainset_loader_no_aug, encoder, k=5)

    retrieval(trainset_loader_no_aug, no_generated_encoder, k=5)
