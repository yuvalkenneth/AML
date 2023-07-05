from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from VICreg_questions import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    freeze_model(encoder)

    # no_neighbors_encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    # no_neighbors_encoder.load_state_dict(torch.load('no_generated_encoder.pth', map_location=device))
    # freeze_model(no_neighbors_encoder)

    # Clustering
    cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=transforms.ToTensor())

    cifar10_trainset = TestingDataset(cifar10_trainset.data, cifar10_trainset.targets,
                                      transform=augmentations.test_transform)

    cifar10_trainset_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=BATCH_SIZE,
                                                          shuffle=False)
    no_neighbors_encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    no_generated_projector = Projector(D=ENCODER_DIM, proj_dim=PROJ_DIM).to(device)
    if os.path.exists('no_generated_encoder.pth'):
        no_neighbors_encoder.load_state_dict(torch.load('no_generated_encoder.pth', map_location=device))
    else:
        nearest_neighbors_set = get_nearest_neighbors(cifar10_trainset_loader, 3, encoder)
        nearest_neighbor_loader = torch.utils.data.DataLoader(nearest_neighbors_set, batch_size=BATCH_SIZE,
                                                              shuffle=True,
                                                              num_workers=4)
        no_generated_optim = torch.optim.Adam(
            list(no_neighbors_encoder.parameters()) + list(no_generated_projector.parameters()), lr=LR,
            betas=BETAS, weight_decay=WEIGHT_DECAY)
        train_no_generated_neighbors(no_neighbors_encoder, no_generated_projector, nearest_neighbor_loader,
                                     no_generated_optim, device)
        torch.save(no_neighbors_encoder.state_dict(), 'no_generated_encoder.pth')
    freeze_model(no_neighbors_encoder)

    all_embeddings, all_labels = get_encoding(encoder, cifar10_trainset_loader)
    all_embeddings_no_neighbors, _ = get_encoding(no_neighbors_encoder, cifar10_trainset_loader)
    all_embeddings = all_embeddings.cpu().detach().numpy()
    all_embeddings_no_neighbors = all_embeddings_no_neighbors.cpu().detach().numpy()

    tsne = TSNE()

    kmeans_data = KMeans(n_clusters=10).fit(all_embeddings)
    kmeans_no_neighbors_data = KMeans(n_clusters=10).fit(all_embeddings_no_neighbors)

    kmeans = kmeans_data.predict(all_embeddings)
    kmeans_no_neighbors = kmeans_no_neighbors_data.predict(all_embeddings_no_neighbors)

    kmeans_centers = kmeans_data.cluster_centers_
    kmeans_no_neighbors_centers = kmeans_no_neighbors_data.cluster_centers_

    combined_data = np.vstack([all_embeddings, kmeans_centers])
    combined_data_no_neighbors = np.vstack([all_embeddings_no_neighbors, kmeans_no_neighbors_centers])

    tsne_combined_data = tsne.fit_transform(combined_data)
    tsne_combined_data_no_neighbors = tsne.fit_transform(combined_data_no_neighbors)

    tsne_data = tsne_combined_data[:-len(kmeans_centers)]
    tsne_kmeans_centers = tsne_combined_data[-len(kmeans_centers):]

    tsne_data_no_neighbors = tsne_combined_data_no_neighbors[:-len(kmeans_no_neighbors_centers)]
    tsne_kmeans_no_neighbors_centers = tsne_combined_data_no_neighbors[-len(kmeans_no_neighbors_centers):]

    # plotting

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=kmeans, ax=ax[0], palette="colorblind")
    sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=all_labels, ax=ax[1], palette="colorblind")

    ax[0].scatter(tsne_kmeans_centers[:, 0], tsne_kmeans_centers[:, 1], c='black', s=100, alpha=0.5)
    ax[1].scatter(tsne_kmeans_centers[:, 0], tsne_kmeans_centers[:, 1], c='black', s=100, alpha=0.5)
    ax[0].set_title('KMeans Clustering')
    ax[1].set_title('True Labels')
    plt.suptitle("Normal VICReg")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(x=tsne_data_no_neighbors[:, 0], y=tsne_data_no_neighbors[:, 1], hue=kmeans_no_neighbors,
                    ax=ax[0], palette="colorblind")
    sns.scatterplot(x=tsne_data_no_neighbors[:, 0], y=tsne_data_no_neighbors[:, 1],
                    hue=all_labels, ax=ax[1], palette="colorblind")
    ax[0].scatter(tsne_kmeans_no_neighbors_centers[:, 0], tsne_kmeans_no_neighbors_centers[:, 1], c='black',
                  s=100, alpha=0.5)
    ax[1].scatter(tsne_kmeans_no_neighbors_centers[:, 0], tsne_kmeans_no_neighbors_centers[:, 1], c='black',
                  s=100, alpha=0.5)
    ax[0].set_title('KMeans Clustering')
    ax[1].set_title('True Labels')
    plt.suptitle("No Generated Neighbors")
    plt.show()

    # Silhouette Score

    print('Silhouette Score Normal VICReg: {}'.format(silhouette_score(all_embeddings, kmeans)))
    print('Silhouette Score without Generated Neighbors: {}'.format(
        silhouette_score(all_embeddings_no_neighbors,
                         kmeans_no_neighbors)))
