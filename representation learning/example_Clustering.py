from clustering_questions import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    freeze_model(encoder)

    # Clustering
    cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=transforms.ToTensor())

    cifar10_trainset = TestingDataset(cifar10_trainset.data, cifar10_trainset.targets,
                                      transform=augmentations.test_transform)

    cifar10_trainset_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=BATCH_SIZE,
                                                          shuffle=False)

    all_embeddings, all_labels = get_encoding(encoder, cifar10_trainset_loader)
    all_embeddings = all_embeddings.cpu().detach().numpy()

    tsne = TSNE()

    kmeans_data = KMeans(n_clusters=10).fit(all_embeddings)

    kmeans = kmeans_data.predict(all_embeddings)

    kmeans_centers = kmeans_data.cluster_centers_

    combined_data = np.vstack([all_embeddings, kmeans_centers])

    tsne_combined_data = tsne.fit_transform(combined_data)

    tsne_data = tsne_combined_data[:-len(kmeans_centers)]
    tsne_kmeans_centers = tsne_combined_data[-len(kmeans_centers):]

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

    # Silhouette Score

    print('Silhouette Score Normal VICReg: {}'.format(silhouette_score(all_embeddings, kmeans)))
