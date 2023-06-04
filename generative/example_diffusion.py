import matplotlib.pyplot as plt
import seaborn as sns

from diffusion_models import *

SAMPLE_SIZE = 3000
T = 1000


def condition(coordinate):
    for i, value in enumerate(np.arange(-0.6, 1.4, 0.4)):
        if coordinate[0] <= value:
            return i


if __name__ == '__main__':
    # UNCONDITIONAL
    data = TrainSet(-1, 1, 2, SAMPLE_SIZE)
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    unconditional_model = Denoiser()
    loss = train_diffusion(unconditional_model, data_loader, EPOCHS, BATCH_SIZE, 2, sigma_t)

    plt.plot(loss)
    plt.title('Loss over Batches')
    plt.show()
    unconditional_model.eval()
    for param in unconditional_model.parameters():
        param.requires_grad = False
    points = point_sampling(unconditional_model, -1 / T, sigma_t, 2, num_samples=1000)[0]
    x, y = zip(*points.detach().tolist())
    sns.scatterplot(x=x, y=y)
    plt.title('Sampling example')
    plt.show()
    noise = torch.randn(1, 2) + torch.tensor([5., 6.5])
    # far_point = point_sampling(unconditional_model, -1 / T, sigma_t, 2, num_samples=1, noise=noise)[0]
    # print(1)

    # CONDITIONAL
    conditional_data = TrainSet(-1, 1, 2, SAMPLE_SIZE, conditional=condition)
    conditional_data_loader = torch.utils.data.DataLoader(conditional_data, batch_size=BATCH_SIZE,
                                                          shuffle=True)

    conditional_model = ConditionedDenoiser(5)
    conditional_loss = train_conditional_diffusion(conditional_model, conditional_data_loader, EPOCHS,
                                                   BATCH_SIZE, 2, sigma_t)

    plt.plot(conditional_loss)
    plt.title('Conditional Loss over Batches')
    plt.show()
    conditional_model.eval()
    for param in conditional_model.parameters():
        param.requires_grad = False
    conditioned_samples = []
    x, y = [], []
    classes = []
    for i in range(5):
        conditioned_samples.append(point_sampling(conditional_model, -1 / T, sigma_t, 2, num_samples=1000,
                                                  seed=i, classes=torch.tensor(i))[0])
        [x.append(sample.detach().tolist()[0]) for sample in conditioned_samples[i]]
        [y.append(sample.detach().tolist()[1]) for sample in conditioned_samples[i]]
        classes.extend([i] * 1000)

    sns.scatterplot(x=x, y=y, hue=classes, palette="Paired")
    plt.title('Conditional Sampling example')
    plt.show()
