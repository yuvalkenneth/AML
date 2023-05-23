import matplotlib.pyplot as plt
import seaborn as sns

from diffusion_models import *


def draw_scatter_plot(x_coordinates, y_coordinates, hue, title):
    sns.scatterplot(x=x_coordinates, y=y_coordinates, hue=hue).set(title=title)
    plt.legend(bbox_to_anchor=(1, 1), title='Class')
    plt.show()


if __name__ == '__main__':
    ### Q1
    # time_steps = np.arange(0, 1, 1 / T)
    # point_progress(torch.tensor([0.5, 0.5]), sigma_t, time_steps)

    ### Q2
    # data = TrainSet(-1, 1, 2, SAMPLE_SIZE)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    # model = Denoiser()
    # loss = train_diffusion(model, data_loader, EPOCHS, BATCH_SIZE, 2, sigma_t)
    # plt.plot(loss)
    # plt.show()
    # point, trajectory = point_sampling(model, -1 / T, sigma_t, 2)

    ### Q3
    # fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # for i, ax in enumerate(fig.axes):
    #     point, trajectory = point_sampling(model, -1 / T, sigma_t, 2, num_samples=1000, seed=i)
    #     ax.scatter(point.detach().numpy()[:, 0], point.detach().numpy()[:, 1], alpha=0.3)
    # plt.show()

    ### Q4
    # points = []
    # T_s = list(range(100, 1500, 100))
    # for t in T_s:
    #     points.append(point_sampling(model, t if t == 0 else -1 / t, sigma_t, 2, num_samples=1)[0].detach().
    #                   numpy().tolist()[0])
    # x, y = zip(*points)
    # sns.scatterplot(x=x, y=y, hue=T_s)
    # plt.show()

    ### Q5

    ### Q6

    #######CONDITIONAL########
    def condition(coordinate):
        for i, value in enumerate(np.arange(-0.6, 1.4, 0.4)):
            if coordinate[0] <= value:
                return i


    conditional_data = TrainSet(-1, 1, 2, SAMPLE_SIZE, conditional=condition)
    conditional_data_loader = torch.utils.data.DataLoader(conditional_data, batch_size=BATCH_SIZE,
                                                          shuffle=True)
    ### Q1
    x, y = zip(*conditional_data.points)
    draw_scatter_plot(x, y, conditional_data.classes, 'Conditional Data')

    conditional_model = ConditionedDenoiser(5)
    conditional_loss = train_conditional_diffusion(conditional_model, conditional_data_loader, EPOCHS,
                                                   BATCH_SIZE, 2, sigma_t)

    ### Q2

    ### Q3
    conditioned_trajectories = []
    x, y, classes = [], [], []
    for i in range(5):
        conditioned_trajectories.append(
            point_sampling(conditional_model, -1 / T, sigma_t, 2, num_samples=1, seed=i, classes=
            torch.tensor(i))[1])
        [x.append(sample.detach().tolist()[0][0]) for sample in conditioned_trajectories[i]]
        [y.append(sample.detach().tolist()[0][1]) for sample in conditioned_trajectories[i]]
        classes.extend([i] * (T + 1))
    draw_scatter_plot(x, y, classes, 'Conditional Trajectories by Class')
    ### Q4
    conditioned_samples = []
    x, y = [], []
    classes = []
    for i in range(5):
        conditioned_samples.append(point_sampling(conditional_model, -1 / T, sigma_t, 2, num_samples=1000,
                                                  seed=i, classes=torch.tensor(i))[0])
        [x.append(sample.detach().tolist()[0]) for sample in conditioned_samples[i]]
        [y.append(sample.detach().tolist()[1]) for sample in conditioned_samples[i]]
        classes.extend([i] * 1000)

    draw_scatter_plot(x, y, classes, 'Conditional Samples by Class')
