import matplotlib.pyplot as plt
import seaborn as sns
import torch


from diffusion_models import *


def draw_scatter_plot(x_coordinates, y_coordinates, hue, title):
    sns.scatterplot(x=x_coordinates, y=y_coordinates, hue=hue).set(title=title)
    plt.legend(bbox_to_anchor=(1, 1), title='Class')
    plt.show()


if __name__ == '__main__':
    # ## Q1
    # time_steps = np.arange(0, 1, 1 / T)
    # point_progress(torch.tensor([0., 0.]), sigma_t, time_steps)
    #
    # ## Q2
    # data = TrainSet(-1, 1, 2, SAMPLE_SIZE)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    # model = Denoiser()
    # loss = train_diffusion(model, data_loader, EPOCHS, BATCH_SIZE, 2, sigma_t)
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False
    # plt.plot(loss)
    # plt.title('Loss over Batches - Q2.2.2.2')
    # plt.show()
    # # point, trajectory = point_sampling(model, -1 / T, sigma_t, 2)
    #
    # ## Q3
    # fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # for i, ax in enumerate(fig.axes):
    #     point, trajectory = point_sampling(model, -1 / T, sigma_t, 2, num_samples=1000, seed=i)
    #     ax.scatter(point.detach().numpy()[:, 0], point.detach().numpy()[:, 1], alpha=0.3)
    # plt.suptitle('Point Sampling - Q2.2.2.3')
    # plt.show()
    #
    # # Q4
    # points = []
    # T_s = list(range(100, 1000, 100))
    # for t in T_s:
    #     points.append(point_sampling(model, t if t == 0 else -1 / t, sigma_t, 2, num_samples=1,
    #                                  seed=0)[0].detach().
    #                   numpy().tolist()[0])
    # x, y = zip(*points)
    # draw_scatter_plot(x, y, T_s, 'Point Sampling as a function of T - Q2.2.2.4')
    #
    # ## Q5
    # fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    # normal_scheduler_points = point_sampling(model, -1 / T, sigma_t, 2, num_samples=1000, seed=0)[0]
    #
    # new_scheduler_points = point_sampling(model, -1 / T, lambda s: s, 2, num_samples=1000, seed=0)[0]
    # x, y = zip(*normal_scheduler_points.detach().numpy().tolist())
    # axs[0, 0].scatter(x, y, alpha=0.3)
    # axs[0, 0].set_title('exp(5(t-1)) Scheduler')
    # x, y = zip(*new_scheduler_points.detach().numpy().tolist())
    # axs[0, 1].scatter(x, y, alpha=0.3)
    # axs[0, 1].set_title('t Scheduler')
    # sqrt_points = point_sampling(model, -1 / T, lambda s: s ** 0.5, 2, num_samples=1000, seed=0)[0]
    # x, y = zip(*sqrt_points.detach().numpy().tolist())
    # axs[1, 0].scatter(x, y, alpha=0.3)
    # axs[1, 0].set_title('sqrt(t) Scheduler')
    #
    # e_15points = point_sampling(model, -1 / T, lambda s: torch.exp(15 * (s - 1)), 2, num_samples=1000,
    #                             seed=0)[0]
    # x, y = zip(*e_15points.detach().numpy().tolist())
    # axs[1, 1].scatter(x, y, alpha=0.3)
    # axs[1, 1].set_title('exp(15(t-1)) Scheduler')
    # plt.suptitle('Point Sampling with different Schedulers - Q2.2.2.5')
    # plt.show()
    #
    #
    # ## Q6
    # denoised_points = []
    # stochastic_points = []
    # trajectories = []
    # noise = torch.randn(1, 2, dtype=model.ln1.weight.dtype)
    # for i in range(10):
    #     denoised_points.append(point_sampling(model, -1 / T, sigma_t, 2, noise=noise)[0])
    # for i in range(10):
    #     points, trajectory = point_sampling(model, -1 / T, sigma_t, 2, num_samples=1, seed=i, noise=noise,
    #                                         stochastic=True)
    #     stochastic_points.append(points)
    #     trajectories.append(trajectory)
    # for i in range(4):
    #     x, y = zip(*trajectories[i])
    #     plt.scatter(x, y, alpha=0.3, label=f'{i}')
    # plt.title('Trajectories - Q2.2.2.6')
    # plt.legend()
    # plt.show()

    # #######CONDITIONAL########
    def condition(coordinate):
        for i, value in enumerate(np.arange(-0.6, 1.4, 0.4)):
            if coordinate[0] <= value:
                return i


    conditional_data = TrainSet(-1, 1, 2, SAMPLE_SIZE, conditional=condition)
    conditional_data_loader = torch.utils.data.DataLoader(conditional_data, batch_size=BATCH_SIZE,
                                                          shuffle=True)
    ### Q1
    x, y = zip(*conditional_data.points)
    draw_scatter_plot(x, y, conditional_data.classes, 'Conditional Data - 2.2.2.1 (conditional)')

    conditional_model = ConditionedDenoiser(5)
    conditional_loss = train_conditional_diffusion(conditional_model, conditional_data_loader, EPOCHS,
                                                   BATCH_SIZE, 2, sigma_t)
    plt.plot(conditional_loss)
    plt.title('Conditional Loss over Batches')
    plt.show()

    ### Q2 - verbal question

    ### Q3
    conditioned_trajectories = []
    x, y, classes = [], [], []
    for i in range(5):
        conditioned_trajectories.append(
            point_sampling(conditional_model, -1 / T, sigma_t, 2, num_samples=1, seed=i, classes=
            torch.tensor(i))[1])
        [x.append(sample[0]) for sample in conditioned_trajectories[i]]
        [y.append(sample[1]) for sample in conditioned_trajectories[i]]
        classes.extend([i] * (T + 1))
    draw_scatter_plot(x, y, classes, 'Conditional Trajectories by Class - Q2.2.2.3 (conditional)')
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

    draw_scatter_plot(x, y, classes, 'Conditional Samples by Class - Q2.2.2.4 (conditional)')

    ### Q5 - verbal question

    ### Q6

    points = [[-0.8, 0.8, 0], [-0.8, 0.8, 2], [-2, 0, 0], [0.6, 0, 3], [0.6, 0, 4]]
    probabilities = []

    for point in points:
        coordinates, label = torch.tensor(point[:2], dtype=torch.float32), torch.tensor([point[2]])
        probabilities.append(point_estimation(coordinates, sigma_t, 1 / T, conditional_model,
                                              label))

    print(f'Probabilities: {probabilities}')
