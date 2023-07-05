from sklearn.metrics import roc_curve, roc_auc_score

from VICreg_questions import *


class MnistAugmented(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


def inverse_density_score(enc, train, normal, anomaly, k=2,
                          device="cuda:0" if torch.cuda.is_available() else "cpu"):
    train_embeddings = get_encoding(enc, train, device)[0]
    normal_embeddings = get_encoding(enc, normal, device)[0]
    anomaly_embeddings = get_encoding(enc, anomaly, device)[0]
    normal_scores = get_distances(normal_embeddings, train_embeddings, k)
    anomaly_scores = get_distances(anomaly_embeddings, train_embeddings, k)
    return normal_scores, anomaly_scores


def get_distances(data, target, k):
    dist_matrix = torch.cdist(data, target)
    values, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
    return values.mean(dim=1)


def run_qualitaive_evaluation():
    _, anomalies_vicreg = torch.topk(torch.cat((cifar10_scores, mnist_scores)), 7, largest=True)
    _, anomalies_no_generated = torch.topk(
        torch.cat((cifar10_scores_no_neighbors, mnist_scores_no_neighbors)), 7, largest=True)
    anomalies_vicreg = torch.flip(anomalies_vicreg, [0])
    anomalies_no_generated = torch.flip(anomalies_no_generated, [0])
    fig, axs = plt.subplots(2, 7)
    fig.suptitle('Anomalies')
    for i in range(7):
        image = cifar10_testset[anomalies_vicreg[i]][0] if anomalies_vicreg[i] < 10000 else \
            mnist_testset[anomalies_vicreg[i] - 10000][0]
        axs[0, i].imshow(image.permute(1, 2, 0))
        axs[0, i].set_title('vicreg {}'.format(i + 1), fontsize=10)
        axs[0, i].axis('off')

        image_no_generated = cifar10_testset[anomalies_no_generated[i]][0] if anomalies_no_generated[
                                                                                  i] < 10000 else \
            mnist_testset[anomalies_no_generated[i] - 10000][0]
        axs[1, i].imshow(image_no_generated.permute(1, 2, 0))
        axs[1, i].set_title('no \ngenerated\n {}'.format(i + 1), fontsize=10)
        axs[1, i].axis('off')
    row_titles = ['Original VICReg', 'No Generated Neighbors']
    for i, ax_row in enumerate(axs):
        ax_row[0].set_ylabel(row_titles[i], rotation=0, size='large', labelpad=20)
    plt.show()


if __name__ == '__main__':
    mnist_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.247, 0.243, 0.261)),
                                          transforms.Resize(32)])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    freeze_model(encoder)

    # Anomaly Detection

    cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=transforms.ToTensor())
    cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                   transform=transforms.ToTensor())
    mnist_testset = torchvision.datasets.MNIST('./data', train=False, download=True,
                                               transform=mnist_transform)

    cifar10_trainset = TestingDataset(cifar10_trainset.data, cifar10_trainset.targets,
                                      transform=augmentations.test_transform)
    cifar10_testset = TestingDataset(cifar10_testset.data, cifar10_testset.targets,
                                     transform=augmentations.test_transform)

    # mnist_aug_testset = MnistAugmented(mnist_testset, augmentations.test_transform)

    cifar10_trainset_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=BATCH_SIZE,
                                                          shuffle=False)

    cifar10_testset_loader = torch.utils.data.DataLoader(cifar10_testset,
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=False)

    mnist_testset_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE,
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

    # Q1 Anomaly Detection - KNN (Inverse) Density Estimation
    cifar10_scores, mnist_scores = inverse_density_score(encoder, cifar10_trainset_loader,
                                                         cifar10_testset_loader, mnist_testset_loader, k=2)
    cifar10_scores_no_neighbors, mnist_scores_no_neighbors = inverse_density_score(no_neighbors_encoder,
                                                                                   cifar10_trainset_loader,
                                                                                   cifar10_testset_loader,
                                                                                   mnist_testset_loader, k=2)

    all_scores = torch.cat((cifar10_scores, mnist_scores))
    all_scores_no_neighbors = torch.cat((cifar10_scores_no_neighbors, mnist_scores_no_neighbors))
    y_true = ([0] * 10000) + ([1] * 10000)

    # Q2 ROC AUC Evaluation
    fpr1, tpr1, thresholds1 = roc_curve(y_true, all_scores.cpu())
    auc1 = roc_auc_score(y_true, all_scores.cpu())

    fpr2, tpr2, thresholds2 = roc_curve(y_true, all_scores_no_neighbors.cpu())
    auc2 = roc_auc_score(y_true, all_scores_no_neighbors.cpu())

    plt.plot(fpr1, tpr1, color="green", label='ROC Curve original VICReg (AUC = {:.2f})'.format(auc1))
    plt.plot(fpr2, tpr2, color="black", label='ROC Curve no generated neighbors (AUC = {:.2f})'.format(auc2))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves and Random Plot - Anomaly Detection')
    plt.legend()
    plt.show()

    # Q3 Qualitative Evaluation - Ambiguity of Anomaly Detection
    run_qualitaive_evaluation()
