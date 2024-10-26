exec(open("yourpath/BNP_DRO_functions.py").read())
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets

df = pd.concat([X, y], axis=1)

n, d = df.shape
y = (df.iloc[:, -1] - df.iloc[:, -1].mean()) / df.iloc[:, -1].std()
df = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
df = pd.concat([y, df], axis=1)

np.random.seed(1234)
df = df.sample(frac=1).reset_index(drop=True)
df_train = df.loc[0:299, :]
df_test = df.loc[300:, :]

n_folds = 10
folds = {}

for k in range(0,n_folds):
    folds[k] = df_train.loc[k*30 : (k+1)*30-1,:]
    
loss_fun = 'squared'
alpha_grid = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
beta = 1
approx_type = 'dirichlet_multinomial'
N_mc = 200
T_steps = 100

# SGD parameters
n_steps = 100000
step_size0 = 100
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
        y = data.iloc[:,0]
        X = data.iloc[:,1:]
        model_l1 = linear_model.Lasso(alpha = alpha, fit_intercept=False)
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
    folds[k] = df_train.loc[k*30 : (k+1)*30-1,:]

oos_lasso_opt = {}
oos_DP_opt = {}
oos_unreg ={}
for k in range(0,n_folds):
    data = folds[k]
    y = data.iloc[:,0]
    X = data.iloc[:,1:]
    model_l1 = linear_model.Lasso(alpha = alpha_opt_lasso, fit_intercept=False)
    model_l1.fit(X, y)
    theta_lasso = model_l1.coef_.T
    oos_lasso_opt[k] = oos_performance(loss_fun = loss_fun, theta = theta_lasso, tst_sample = df_test.values)
    model_unreg = linear_model.LinearRegression(fit_intercept=False)
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

print(np.mean(list(oos_unreg.values())))
print(np.std(list(oos_unreg.values())))
print(np.mean(list(oos_lasso_opt.values())))
print(np.std(list(oos_lasso_opt.values())))
print(np.mean(list(oos_DP_opt.values())))
print(np.std(list(oos_DP_opt.values())))