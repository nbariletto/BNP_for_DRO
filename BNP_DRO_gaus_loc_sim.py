##################################################
# Gaussian Mean Estimation Simulation Experiment #
##################################################

exec(open("yourpath/BNP_DRO_functions.py").read())


# Data parameters
n_sims = 100
n_train = 10
n_test = 5000
n_tot = n_train + n_test
n_out = 3  # number of outlier observations
mn_true, mn_out = 0, 5
s = 1

# Criterion parameters
loss_fun = 'gaussian_loc_lik'
beta_grid = [1, 'inf']
alpha_grid = [1, 2, 5, 10]
approx_type = 'dirichlet_multinomial'
N_mc = 300
T_steps = 50
mn_0 = (n_train * mn_true + n_out * mn_out) / (n_train + n_out)

# SGD parameters
n_steps = 50000
step_size0 = 20
n_passes = int(np.ceil(n_steps / N_mc))

np.random.seed(12345)

BNP_oos = {}
MLE_oos = {}
BNP_theta = {}
MLE_theta = {}

for sim in range(1, n_sims + 1):
    data = np.random.normal(mn_true, s, n_tot)

    trn_sample = data[0:n_train]
    trn_sample = np.concatenate((trn_sample, np.random.normal(mn_out, s, n_out)))
    tst_sample = data[n_train:]

    theta_MLE = trn_sample.mean()
    MLE_oos[(n_train, sim)] = oos_performance(loss_fun=loss_fun, theta=theta_MLE, tst_sample=tst_sample)
    MLE_theta[(n_train, sim)] = theta_MLE
    theta_0 = 0

    for alpha in alpha_grid:
        theta_a_neut = np.concatenate((trn_sample, np.array([mn_0 for i in range(0, alpha)]))).mean()
        a = approx_criterion_stacked(N_mc=N_mc, T_steps=T_steps, approx_type=approx_type, data=trn_sample, alpha=alpha,
                              loss_fun=loss_fun)
        for beta in beta_grid:
            if beta == 'inf':
                BNP_theta[(n_train, sim, alpha, beta)] = theta_a_neut
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun=loss_fun, theta=theta_a_neut,
                                                                        tst_sample=tst_sample)
            else:
                theta_path, values_path = SGD_alternative_stacked(loss_fun=loss_fun,
                                                          theta_0=theta_0, beta=beta, criterion=a,
                                                          n_passes=n_passes, step_size0=step_size0)
                BNP_theta[(n_train, sim, alpha, beta)] = theta_path[-1]
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun=loss_fun, theta=theta_path[-1],
                                                                        tst_sample=tst_sample)

df = pd.DataFrame(columns=['n_train', 'sim', 'alpha', 'beta', 'oos_performance', 'theta_dist_truth'])

indexes = []
for sim in range(1, n_sims + 1):
    for alpha in alpha_grid:
        indexes.append([n_train, sim, alpha, 'MLE'])
        for beta in beta_grid:
            indexes.append([n_train, sim, alpha, beta])

df[['n_train', 'sim', 'alpha', 'beta']] = indexes

for sim in range(1, n_sims + 1):
    df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['beta'] == 'MLE'), 'oos_performance'] = MLE_oos[
        (n_train, sim)]
    df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['beta'] == 'MLE'), 'theta_dist_truth'] = np.abs(
        MLE_theta[(n_train, sim)])
    for alpha in alpha_grid:
        for beta in beta_grid:
            df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['alpha'] == alpha) & (
                        df['beta'] == beta), 'oos_performance'] = BNP_oos[(n_train, sim, alpha, beta)]
            df.loc[(df['n_train'] == n_train) & (df['sim'] == sim) & (df['alpha'] == alpha) & (
                        df['beta'] == beta), 'theta_dist_truth'] = np.abs(BNP_theta[(n_train, sim, alpha, beta)])


# Plot of Results
labels = {'oos_performance': 'Test Mean Neg. Log-Lik.', 'theta_dist_truth': r'Dist. from Truth ($\hat\theta$)'}
alpha_grid = [1, 2, 5, 10]

df.loc[df['beta'] == 1, 'beta'] = 'A. Averse'
df.loc[df['beta'] == 'inf', 'beta'] = 'A. Neutral'

fig, axs = plt.subplots(nrows=len(labels), ncols=len(alpha_grid), figsize=(len(alpha_grid) * 5, len(labels) * 3),
                        sharex='col', sharey='row')

for j, metric in enumerate(labels.keys()):
    axs[j, 0].set_ylabel(labels[metric], fontsize=16)

for j, metric in enumerate(labels.keys()):
    for i, alpha in enumerate(alpha_grid):
        tmp = df.loc[df['alpha'] == alpha,]
        tmp_mean = tmp.groupby('beta')[['oos_performance', 'theta_dist_truth']].mean().reset_index()
        tmp_std = tmp.groupby('beta')[['oos_performance', 'theta_dist_truth']].std().reset_index()
        x = np.arange(3)

        ax = axs[j, i]
        bar1 = ax.bar(x - 0.2, tmp_mean[metric], width=0.4, label='Mean', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(tmp_mean['beta'], fontsize=14)
        ax.tick_params(axis='y', labelsize=14, labelcolor='#1f77b4')

        if j == 0:
            ax.set_title(fr'$\alpha = {alpha}$', fontsize=20)

        ax2 = ax.twinx()
        bar2 = ax2.bar(x + 0.2, tmp_std[metric], width=0.4, label='St. Dev', color='#ff7f0e', zorder=2)
        ax2.tick_params(axis='y', labelsize=14, labelcolor='#ff7f0e')

        if i != len(alpha_grid) - 1:
            ax2.set_yticks([])
            ax2.yaxis.tick_right()

        ax2.grid(False)

        ax.legend().set_visible(False)
        ax2.legend().set_visible(False)

        ax.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
        ax.grid(axis='y', linestyle='-', alpha=0.5, zorder=1)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.4f}'.format(x)))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.4f}'.format(x)))

handles = [bar1, bar2]
labels = ['Mean', 'St. Dev']
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.001, 0.5), fontsize=16)

plt.tight_layout()
plt.show()