import matplotlib.pyplot as plt
import seaborn as sns

from diffusion_models import *


def draw_scatter_plot(x_coordinates, y_coordinates, hue, title, palette='Paired', legend_title='Class',
                      change_lims=True):
    sns.scatterplot(x=x_coordinates, y=y_coordinates, hue=hue, legend="full", palette=palette).set(
        title=title)
    plt.legend(bbox_to_anchor=(1, 1), title=legend_title)
    if change_lims:
        plt.ylim(min(y_coordinates) - 0.1, max(y_coordinates) + 0.1)
        plt.xlim(min(x_coordinates) - 0.1, max(x_coordinates) + 0.1)
    plt.show()


def point_progress(x_, noise_scheduler, steps):
    point_in_space = []

    for ind, t in enumerate(steps):
        noise = torch.randn(1, 2)
        x_ = x_ + noise_scheduler(torch.tensor(t)) * noise * (steps[ind] - (steps[ind - 1] if ind > 0 else 0))
        # point_in_space.append((starting_point + (noise_scheduler(torch.tensor(t)) * noise)[0]).tolist())
        point_in_space.append(x_.tolist()[0])
    x, y = zip(*point_in_space)
    plt.scatter(x=x, y=y, c=[0] + steps, cmap="inferno", alpha=0.3)
    cbar = plt.colorbar()
    cbar.set_label('Time Step')
    plt.title('Point Progression - Q2.2.1 (starting at (0,0))')
    plt.show()

    return point_in_space


def point_sampling_revised(denoiser, step_size, noise_scheduler, dim, target, num_samples=1, noise=None):
    # Bonus question
    z = torch.randn(num_samples, dim, dtype=denoiser.ln1.weight.dtype) if noise is None else noise.detach(
    ).clone()
    trajectory = [z.detach().tolist()[0]]
    if step_size != 0:
        for t in np.arange(1, 0, step_size):
            t = torch.tensor(t, requires_grad=True, dtype=denoiser.ln1.weight.dtype).reshape(1, 1)
            t.retain_grad()
            sigma = noise_scheduler(t)
            sigma_derivative = torch.autograd.grad(sigma, t)[0]
            estimated_noise = denoiser(z, t.expand(num_samples, 1))
            z_hat = z - sigma * estimated_noise
            score = (target - z_hat) / (sigma ** 2)
            dz = -sigma_derivative * sigma * score * step_size
            z = z + dz
            trajectory.append(z.detach().tolist()[0])
    return z, trajectory


if __name__ == '__main__':
    # Q1
    time_steps = np.arange(0, 1 + (1 / T), 1 / T)
    point_progress(torch.tensor([0., 0.]), sigma_t, time_steps)

    # Q2
    data = TrainSet(-1, 1, 2, SAMPLE_SIZE)
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    unconditional_model = Denoiser()
    loss = train_diffusion(unconditional_model, data_loader, EPOCHS, BATCH_SIZE, 2, sigma_t)
    plt.plot(loss)
    plt.title('Loss over Batches - Q2.2.2.2')
    plt.show()
    unconditional_model.eval()
    for param in unconditional_model.parameters():
        param.requires_grad = False

    # Q3
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(fig.axes):
        point, trajectory = point_sampling(unconditional_model, -1 / T, sigma_t, 2, num_samples=1000, seed=i)
        ax.scatter(point.detach().numpy()[:, 0], point.detach().numpy()[:, 1], alpha=0.3)
        ax.set_title(f'Point Sampling - seed = {i}')
    plt.suptitle('Point Sampling - Q2.2.2.3', fontsize=16)
    plt.show()

    # Q4
    points = []
    T_s = [100, 500, 1000, 2000, 5000, 10000, 50000]
    for t in T_s:
        points.append(point_sampling(unconditional_model, t if t == 0 else -1 / t, sigma_t, 2, num_samples=1,
                                     seed=0)[0].detach().
                      numpy().tolist()[0])
    x, y = zip(*points)
    draw_scatter_plot(x, y, T_s, 'Point Sampling as a function of T - Q2.2.2.4', legend_title="T",
                      change_lims=False)

    # Q5
    lin_points = np.linspace(0, 1, 1000)
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig1, axs1 = plt.subplots(2, 2, figsize=(8, 8))
    normal_scheduler_points = \
        point_sampling(unconditional_model, -1 / T, sigma_t, 2, num_samples=1000, seed=0)[0]
    x, y = zip(*normal_scheduler_points.detach().numpy().tolist())
    axs[0, 0].scatter(x, y, alpha=0.3)
    axs[0, 0].set_title('exp(5(t-1)) Scheduler', fontsize=20)
    axs1[0, 0].scatter(lin_points, sigma_t(torch.tensor(lin_points)).detach().numpy())
    axs1[0, 0].set_title('exp(5(t-1))', fontsize=14)

    new_scheduler_points = \
        point_sampling(unconditional_model, -1 / T, lambda s: s, 2, num_samples=1000, seed=0)[0]
    x, y = zip(*new_scheduler_points.detach().numpy().tolist())
    axs[0, 1].scatter(x, y, alpha=0.3)
    axs[0, 1].set_title('t Scheduler', fontsize=20)
    axs1[0, 1].scatter(lin_points, (lambda s: s)(lin_points))
    axs1[0, 1].set_title('t', fontsize=14)

    sqrt_points = \
        point_sampling(unconditional_model, -1 / T, lambda s: s ** 0.5, 2, num_samples=1000, seed=0)[0]
    x, y = zip(*sqrt_points.detach().numpy().tolist())
    axs[1, 0].scatter(x, y, alpha=0.3)
    axs[1, 0].set_title('sqrt(t) Scheduler', fontsize=20)
    axs1[1, 0].scatter(lin_points, (lambda s: s ** 0.5)(lin_points))
    axs1[1, 0].set_title('sqrt(t)', fontsize=14)

    e_15points = \
        point_sampling(unconditional_model, -1 / T, lambda s: torch.exp(15 * (s - 1)), 2, num_samples=1000,
                       seed=0)[0]
    x, y = zip(*e_15points.detach().numpy().tolist())
    axs[1, 1].scatter(x, y, alpha=0.3)
    axs[1, 1].set_title('exp(15(t-1)) Scheduler', fontsize=20)
    axs1[1, 1].scatter(lin_points, torch.exp(15 * (torch.tensor(lin_points) - 1)).detach().numpy())
    axs1[1, 1].set_title('exp(15(t-1))', fontsize=14)

    fig.suptitle('Point Sampling with different Schedulers - Q2.2.2.5', fontsize=25)
    fig1.suptitle('Schedulers - Q2.2.2.5', fontsize=20)
    fig.show()
    fig1.show()

    ## Q6
    denoised_points = []
    stochastic_points = []
    trajectories = []
    noise = torch.randn(1, 2, dtype=unconditional_model.ln1.weight.dtype)
    for i in range(10):
        denoised_points.append(point_sampling(unconditional_model, -1 / T, sigma_t, 2, noise=noise)[0])
    for i in range(10):
        points, trajectory = point_sampling(unconditional_model, -1 / T, sigma_t, 2, num_samples=1, seed=i,
                                            noise=noise,
                                            stochastic=True)
        stochastic_points.append(points)
        trajectories.append(trajectory)
    for i in range(4):
        x, y = zip(*trajectories[i])
        plt.scatter(x, y, alpha=0.3, label=f'{i}')
    plt.title('Trajectories - Q2.2.2.6')
    plt.legend()
    plt.show()


    #######CONDITIONAL########

    def condition(coordinate):
        for i, value in enumerate(np.arange(-0.6, 1.4, 0.4)):
            if coordinate[0] <= value:
                return i


    conditional_data = TrainSet(-1, 1, 2, SAMPLE_SIZE, conditional=condition)
    conditional_data_loader = torch.utils.data.DataLoader(conditional_data, batch_size=BATCH_SIZE,
                                                          shuffle=True)
    # Q1
    x, y = zip(*conditional_data.points)
    draw_scatter_plot(x, y, conditional_data.classes, 'Conditional Data - 2.2.2.1 (conditional)')

    conditional_model = ConditionedDenoiser(5)
    conditional_loss = train_conditional_diffusion(conditional_model, conditional_data_loader, EPOCHS,
                                                   BATCH_SIZE, 2, sigma_t)
    plt.plot(conditional_loss)
    plt.title('Conditional Loss over Batches')
    plt.show()
    conditional_model.eval()
    for param in conditional_model.parameters():
        param.requires_grad = False

    # Q2 - verbal question

    # Q3
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
    # Q4
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

    # Q5 - verbal question

    # Q6

    points = [[-0.8, 0.8, 0], [-0.8, 0.8, 2], [-2, 0, 0], [0.5, 0, 3], [0.9, 0.5, 4]]
    log_probabilities = []

    for point in points:
        coordinates, label = torch.tensor(point[:2], dtype=torch.float32), torch.tensor([point[2]])
        log_probabilities.append(point_estimation(coordinates, sigma_t, 1 / T, conditional_model,
                                                  label))
    print(f"points: {points}")
    print(f'ELBO: {log_probabilities}')
    x, y = zip(*conditional_data.points)
    sns.scatterplot(x=x, y=y, hue=conditional_data.classes, palette='Paired', alpha=0.05, s=100)
    z = sns.scatterplot(x=[point[0] for point in points], y=[point[1] for point in points], hue=[point[2] for
                                                                                                 point in
                                                                                                 points],
                        marker='x', s=100, zorder=10, legend=False, alpha=1, palette='Paired')
    plt.ylim(min(y) - 0.1, max(y) + 0.1)
    plt.xlim(-2.1, max(x) + 0.1)
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., title='Class')
    plt.show()

    # Bonus Question
    bonus_output = point_sampling_revised(unconditional_model, -1 / T, lambda s: torch.exp(15 * (s - 1)), 2,
                                          num_samples=1, target=torch.tensor([4., 5.5]))
    print(f'bonus output: {bonus_output[0]}')
    x, y = zip(*bonus_output[1])
    plt.scatter(x, y, cmap="viridis", c=list(np.arange(1, 0 - 1 / T, -1 / T)))
    cbar = plt.colorbar()
    cbar.set_label('TIME')
    plt.title('Bonus Question')
    plt.show()
