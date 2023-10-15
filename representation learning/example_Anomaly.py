from anomaly_questions import *

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

    cifar10_trainset_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=BATCH_SIZE,
                                                          shuffle=False)

    cifar10_testset_loader = torch.utils.data.DataLoader(cifar10_testset,
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=False)

    mnist_testset_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE,
                                                       shuffle=False)

    # Q1 Anomaly Detection - KNN (Inverse) Density Estimation
    cifar10_scores, mnist_scores = inverse_density_score(encoder, cifar10_trainset_loader,
                                                         cifar10_testset_loader, mnist_testset_loader, k=2)

    all_scores = torch.cat((cifar10_scores, mnist_scores))
    y_true = ([0] * 10000) + ([1] * 10000)

    # Q2 ROC AUC Evaluation
    fpr1, tpr1, thresholds1 = roc_curve(y_true, all_scores.cpu())
    auc1 = roc_auc_score(y_true, all_scores.cpu())

    plt.plot(fpr1, tpr1, color="green", label='ROC Curve original VICReg (AUC = {:.2f})'.format(auc1))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves and Random Plot - Anomaly Detection')
    plt.legend()
    plt.show()
