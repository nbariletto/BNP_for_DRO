#######################
# Diabetes Experiment #
#######################


exec(open("yourpath/BNP_DRO_functions.py").read())

# Data downloaded from https://www.kaggle.com/datasets/kandij/diabetes-dataset?resource=download

df = pd.read_csv('diabetes2.csv')
n, d = df.shape
df.loc[df['Outcome']==0,'Outcome'] = -1
y = df.iloc[:, -1]  # Extracting the last column
df = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
df = pd.concat([y, df], axis=1)

np.random.seed(12345)
df = df.sample(frac=1).reset_index(drop=True)
df_train = df.loc[0:299, :]
df_test = df.loc[300:, :]

n_folds = 15
folds = {}

for k in range(0,n_folds):
    folds[k] = df_train.loc[k*20 : (k+1)*20-1,:]
    
loss_fun = 'logistic'
alpha_grid = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
beta = 1
approx_type = 'dirichlet_multinomial'
N_mc = 200
T_steps = 100

# SGD parameters
n_steps = 300000
step_size0 = 1000
n_passes = int(np.ceil(n_steps/N_mc))
oos_lasso = {}
oos_DP = {}
theta_0 = np.zeros(d-1)

for alpha in alpha_grid:
    lasso_oos = 0
    DP_oos = 0
    for k in range(0,n_folds):
        data = folds[k]
        tmp_test = pd.concat([v for j, v in folds.items() if j != k], axis=0)
        y = data['Outcome']
        X = data.iloc[:,1:]
        model_l1 = LogisticRegression(penalty='l1', C = 1/alpha, solver='liblinear', fit_intercept=False)
        model_l1.fit(X, y)
        theta_lasso = model_l1.coef_.T
        lasso_oos += oos_performance(loss_fun = loss_fun, theta = theta_lasso, tst_sample = tmp_test.values)
        a = approx_criterion_stacked(N_mc = N_mc, T_steps = T_steps, approx_type = approx_type, data = data.values, alpha = alpha, loss_fun = loss_fun)
        theta_path, values_path = SGD_alternative_stacked(loss_fun = loss_fun,
                                                          theta_0 = theta_0, beta = beta, criterion = a,
                                                          n_passes = n_passes, step_size0 = step_size0)
        theta_DP = theta_path[-1]
        DP_oos += oos_performance(loss_fun = loss_fun, theta = theta_DP, tst_sample = tmp_test.values)
    oos_lasso[alpha] = lasso_oos/n_folds
    oos_DP[alpha] = DP_oos/n_folds

print(oos_lasso)
print(oos_DP)
alpha_opt_lasso = min(oos_lasso, key=oos_lasso.get)
alpha_opt_DP = min(oos_DP, key=oos_DP.get)
df_train = df_train.sample(frac=1).reset_index(drop=True)
folds = {}

for k in range(0,n_folds):
    folds[k] = df_train.loc[k*20 : (k+1)*20-1,:]

oos_lasso_opt = {}
oos_DP_opt = {}
oos_unreg ={}
for k in range(0,n_folds):
    data = folds[k]
    y = data['Outcome']
    X = data.iloc[:,1:]
    model_l1 = LogisticRegression(penalty='l1', C = 1/alpha_opt_lasso, solver='liblinear', fit_intercept=False)
    model_l1.fit(X, y)
    theta_lasso = model_l1.coef_.T
    oos_lasso_opt[k] = oos_performance(loss_fun = loss_fun, theta = theta_lasso, tst_sample = df_test.values)
    model_unreg = LogisticRegression(penalty=None, fit_intercept=False)
    model_unreg.fit(X, y)
    theta_unreg = model_unreg.coef_.T
    oos_unreg[k] = oos_performance(loss_fun = loss_fun, theta = theta_unreg, tst_sample = df_test.values)
    a = approx_criterion_stacked(N_mc = N_mc, T_steps = T_steps, approx_type = approx_type, data = data.values,
                                 alpha = alpha_opt_DP, loss_fun = loss_fun)
    theta_path, values_path = SGD_alternative_stacked(loss_fun = loss_fun,
                                                      theta_0 = theta_0, beta = beta, criterion = a,
                                                      n_passes = n_passes, step_size0 = step_size0)
    theta_DP = theta_path[-1]
    oos_DP_opt[k] = oos_performance(loss_fun = loss_fun, theta = theta_DP, tst_sample = df_test.values)

print(np.mean([value for value in oos_unreg.values() if value != float('inf')]))
print(np.std([value for value in oos_unreg.values() if value != float('inf')]))
print(np.mean(list(oos_lasso_opt.values())))
print(np.std(list(oos_lasso_opt.values())))
print(np.mean(list(oos_DP_opt.values())))
print(np.std(list(oos_DP_opt.values())))