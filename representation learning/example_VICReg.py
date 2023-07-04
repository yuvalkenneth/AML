# import augmentations
from VICreg_questions import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    # projector = Projector(D=ENCODER_DIM, proj_dim=PROJ_DIM).to(device)

    # Load weights
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    # projector.load_state_dict(torch.load('projector.pth', map_location=device))

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transforms.ToTensor())

    aug_trainset = TestingDataset(trainset.data, trainset.targets, transform=augmentations.test_transform)
    aug_testset = TestingDataset(testset.data, testset.targets, transform=augmentations.test_transform)

    shuffled_aug_trainset = torch.utils.data.DataLoader(aug_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    trainset_loader = torch.utils.data.DataLoader(aug_trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    testset_loader = torch.utils.data.DataLoader(aug_testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # Linear Probing
    linear_probe = LinearProbe(ENCODER_DIM, 10).to(device)
    probing_acc = linear_probing(train_loader=shuffled_aug_trainset,
                                 test_loader=testset_loader, enc=encoder, probe=linear_probe,
                                 dev=device, path_name="probe_weights.pth")
    print('Linear Probing Accuracy: ', probing_acc)
    classes_indices = get_classes_indices(trainset_loader)
    retrieval(trainset_loader, encoder, k=5, class_indices=classes_indices,
              original_images=trainset)
