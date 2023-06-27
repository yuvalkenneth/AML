from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from VICreg_questions import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    freeze_model(encoder)

    no_neighbors_encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    no_neighbors_encoder.load_state_dict(torch.load('no_generated_encoder.pth', map_location=device))
    freeze_model(no_neighbors_encoder)

    # Clustering
    cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=transforms.ToTensor())

    cifar10_trainset = TestingDataset(cifar10_trainset.data, cifar10_trainset.targets,
                                      transform=augmentations.test_transform)

    cifar10_trainset_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=BATCH_SIZE,
                                                          shuffle=False)

    all_embeddings, all_labels = get_encoding(encoder, cifar10_trainset_loader)
    all_embeddings_no_neighbors, _ = get_encoding(no_neighbors_encoder, cifar10_trainset_loader)
    all_embeddings = all_embeddings.cpu().detach().numpy()
    all_embeddings_no_neighbors = all_embeddings_no_neighbors.cpu().detach().numpy()

    kmeans = KMeans(n_clusters=10).fit_predict(all_embeddings)
    kmeans_no_neighbors = KMeans(n_clusters=10).fit_predict(all_embeddings_no_neighbors)

    tsne_data = TSNE().fit_transform(all_embeddings)
    tsne_data_no_neighbors = TSNE().fit_transform(all_embeddings_no_neighbors)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=kmeans, ax=ax[0], palette="Paired")
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=all_labels, ax=ax[1], palette="Paired")
    ax[0].set_title('KMeans Clustering')
    ax[1].set_title('True Labels')
    plt.suptitle("Normal VICReg")
    plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(x=tsne_data_no_neighbors[:, 0], y=tsne_data_no_neighbors[:, 1], hue=kmeans_no_neighbors,
                    ax=ax[0], palette="Paired")
    sns.scatterplot(x=tsne_data_no_neighbors[:, 0], y=tsne_data_no_neighbors[:, 1],
                    hue=all_labels, ax=ax[1], palette="Paired")
    ax[0].set_title('KMeans Clustering')
    ax[1].set_title('True Labels')
    plt.suptitle("No Generated Neighbors")
    plt.show()

    # Silhouette Score

    print('Silhouette Score Normal VICReg: {}'.format(silhouette_score(all_embeddings, kmeans)))
    print('Silhouette Score without Generated Neighbors: {}'.format(
        silhouette_score(all_embeddings_no_neighbors,
                         kmeans_no_neighbors)))
