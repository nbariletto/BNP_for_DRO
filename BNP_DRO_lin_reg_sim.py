###########################################
# Linear Regression Simulation Experiment #
###########################################

exec(open("yourpath/BNP_DRO_functions.py").read())

# Data parameters
dim = 90
active_params = 5
true_coefs = np.append(np.ones(active_params), np.zeros(dim - active_params))
means = np.zeros(dim)
covariance = 0.3 * np.ones(dim) + (1 - 0.3) * np.eye(dim)
n_train = 100
n_test = 5000
n_tot = n_train + n_test

# Criterion parameters
loss_fun = 'squared'
beta_grid = [1, 'inf']
alpha_grid = [1, 2, 5, 10]
approx_type = 'dirichlet_multinomial'
N_mc = 300
T_steps = 50

# SGD parameters
n_steps = 10000
step_size0 = 50
n_passes = int(np.ceil(n_steps / N_mc))

np.random.seed(1234)

BNP_oos = {}
OLS_oos = {}
BNP_theta = {}
OLS_theta = {}

n_sims = 200
theta_0 = np.zeros(dim)
for sim in range(1, n_sims + 1):
    data = np.random.multivariate_normal(means, covariance, n_tot)
    y = np.sum(data[:, :active_params], axis=1) + np.random.normal(0, 0.5, n_tot)

    data = np.column_stack((y, data))
    trn_sample = data[0:n_train, :]
    tst_sample = data[n_train:, :]

    theta_ols = np.linalg.inv(trn_sample[:, 1:].transpose() @ trn_sample[:, 1:]) @ trn_sample[:,
                                                                                   1:].transpose() @ trn_sample[:, 0]

    OLS_oos[(n_train, sim)] = oos_performance(loss_fun=loss_fun, theta=theta_ols, tst_sample=tst_sample)
    OLS_theta[(n_train, sim)] = theta_ols

    for alpha in [al/n_train for al in alpha_grid]:
        a= approx_criterion_stacked(N_mc=N_mc, T_steps=T_steps, approx_type=approx_type, data=trn_sample, alpha=alpha,
                             loss_fun=loss_fun)
        theta_ridge = np.linalg.inv(
            trn_sample[:, 1:].transpose() @ trn_sample[:, 1:] + (alpha) * np.eye(dim)) @ trn_sample[:,
                                                                                                   1:].transpose() @ trn_sample[
                                                                                                                     :,
                                                                                                                     0]
        for beta in beta_grid:
            if beta == 'inf':
                BNP_theta[(n_train, sim, alpha, beta)] = theta_ridge
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun=loss_fun, theta=theta_ridge,
                                                                       tst_sample=tst_sample)
            else:
                theta_path, values_path = SGD_alternative_stacked(loss_fun=loss_fun,
                                                          theta_0=theta_0, beta=beta, criterion=a,
                                                          n_passes=n_passes, step_size0=step_size0)
                BNP_theta[(n_train, sim, alpha, beta)] = theta_path[-1]
                BNP_oos[(n_train, sim, alpha, beta)] = oos_performance(loss_fun=loss_fun, theta=theta_path[-1],
                                                                       tst_sample=tst_sample)


df1 = pd.DataFrame(columns=['n_train', 'sim', 'alpha', 'beta', 'oos_performance', 'theta_norm', 'theta_dist_truth'])

indexes = []
for sim in range(1, n_sims + 1):
    for alpha in alpha_grid:
        indexes.append([n_train, sim, alpha, 'ols'])
        for beta in beta_grid:
            indexes.append([n_train, sim, alpha, beta])

df1[['n_train', 'sim', 'alpha', 'beta']] = indexes

for sim in range(1, n_sims + 1):
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'oos_performance'] = np.sqrt(
        OLS_oos[(n_train, sim)])
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'theta_norm'] = np.sqrt(
        np.square(OLS_theta[(n_train, sim)]).sum())
    df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['beta'] == 'ols'), 'theta_dist_truth'] = np.sqrt(
        np.square(OLS_theta[(n_train, sim)] - true_coefs).sum())
    for alpha in alpha_grid:
        for beta in beta_grid:
            df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (
                        df1['beta'] == beta), 'oos_performance'] = np.sqrt(BNP_oos[(n_train, sim, alpha/n_train, beta)])
            df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (
                        df1['beta'] == beta), 'theta_norm'] = np.sqrt(
                np.square(BNP_theta[(n_train, sim, alpha/n_train, beta)]).sum())
            df1.loc[(df1['n_train'] == n_train) & (df1['sim'] == sim) & (df1['alpha'] == alpha) & (
                        df1['beta'] == beta), 'theta_dist_truth'] = np.sqrt(
                np.square(BNP_theta[(n_train, sim, alpha/n_train, beta)] - true_coefs).sum())

# Plot of Results
labels = {'oos_performance': 'Test RMSE', 'theta_dist_truth': r'L2 Dist. from Truth ($\hat\theta$)',
          'theta_norm': r'L2 Norm ($\hat\theta$)'}
alpha_grid = [1, 2, 5, 10]

df1.loc[df1['beta'] == 1, 'beta'] = 'A. Averse'
df1.loc[df1['beta'] == 'inf', 'beta'] = 'A. Neutral'
df1.loc[df1['beta'] == 'ols', 'beta'] = 'OLS'

fig, axs = plt.subplots(nrows=len(labels), ncols=len(alpha_grid), figsize=(len(alpha_grid) * 5, len(labels) * 3),
                        sharex='col', sharey='row')

for j, metric in enumerate(labels.keys()):
    axs[j, 0].set_ylabel(labels[metric], fontsize=16)

for j, metric in enumerate(labels.keys()):
    for i, alpha in enumerate(alpha_grid):
        tmp = df1.loc[df1['alpha'] == alpha,]
        tmp_mean = tmp.groupby('beta')[['oos_performance', 'theta_norm', 'theta_dist_truth']].mean().reset_index()
        tmp_std = tmp.groupby('beta')[['oos_performance', 'theta_norm', 'theta_dist_truth']].std().reset_index()
        x = np.arange(3)

        ax = axs[j, i]
        bar1 = ax.bar(x - 0.2, tmp_mean[metric], width=0.4, label='Mean', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(tmp_mean['beta'], fontsize=14)
        ax.tick_params(axis='y', labelsize=14, labelcolor='#1f77b4')

        if j == 0:
            ax.set_title(fr'$a = {alpha}$', fontsize=20)

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

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.4f}'.format(x)))

handles = [bar1, bar2]
labels = ['Mean', 'St. Dev']
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)

plt.tight_layout()
plt.show()