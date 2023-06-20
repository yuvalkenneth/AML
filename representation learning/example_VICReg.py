from VICreg_questions import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(D=ENCODER_DIM, device=device).to(device)
    projector = Projector(D=ENCODER_DIM, proj_dim=PROJ_DIM).to(device)

    # Load weights
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    projector.load_state_dict(torch.load('projector.pth', map_location=device))

    # Linear Probing
    linear_probe = LinearProbe(ENCODER_DIM, 10).to(device)
    probing_acc = linear_probing(encoder_dim=ENCODER_DIM, num_classes=10, train_loader=trainloader_downstream,
                                 test_loader=testloader_downstream, enc=encoder, probe_to_train=linear_probe,
                                 dev=device, path_name="probe_weights.pth")
    print('Linear Probing Accuracy: ', probing_acc)
