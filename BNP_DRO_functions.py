###############
## Functions ##
###############

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import tqdm as tqdm
from matplotlib.ticker import FuncFormatter


def stick_breaking_sampling_stacked(N_MC, T_steps, n, alpha):
    # Sample T_steps + 1 stick-breaking weights
    tmp1 =np.cumprod(np.random.beta(1, alpha + n, size=(N_MC,T_steps)),axis=1)
    tmp2 = 1 - np.sum(tmp1,axis=1,keepdims=True)
    return np.concatenate([tmp1,tmp2],axis=1)

def dirichlet_multinomial_sampling_stacked(N_MC,T_steps, n, alpha):
    # Sample T_steps + 1 Dirichlet weights
    shape = (alpha + n) / (T_steps + 1)
    tmp = np.random.gamma(shape, 1, size=(N_MC,T_steps + 1))
    return tmp / np.sum(tmp,axis=1,keepdims=True)

def atom_sampling(T_steps, data, n, alpha, loss_fun, mn_0=0):
    # Sample atoms according to the DP predictive
    if loss_fun != 'gaussian_loc_lik':
        d = len(data[0, :])
    p = alpha / (alpha + n)
    atoms = []
    for j in range(1, T_steps + 2):
        if np.random.random() < p:
            if loss_fun == 'gaussian_loc_lik':
                atoms.append(np.random.normal(mn_0, 1))
            elif loss_fun == 'squared':
                atoms.append(np.random.normal(0, 1, d))
            elif loss_fun == 'logistic':
                y = np.random.choice([-1, 1])
                X = np.random.normal(0,1,d-1)
                dt = np.append(y, X)
                atoms.append(dt)
            elif loss_fun == 'pinball_median':
                atoms.append(np.random.normal(0, 1, d))
        else:
            if loss_fun == 'gaussian_loc_lik':
                atoms.append(data[np.random.randint(0, n)])
            elif loss_fun == 'squared':
                atoms.append(data[np.random.randint(0, n), :])
            elif loss_fun == 'logistic':
                atoms.append(data[np.random.randint(0,n),:])
            elif loss_fun == 'pinball_median':
                atoms.append(data[np.random.randint(0,n),:])
    return np.array(atoms)

def approx_criterion_stacked(N_mc, T_steps, approx_type, data, alpha, loss_fun, mn_0=0):
    # Wrap weights and atoms together
    if loss_fun == 'gaussian_loc_lik':
        n = len(data)
    else:
        n = len(data[:, 0])
    if approx_type == 'stick_breaking':
        return stick_breaking_sampling_stacked(N_mc,T_steps, n, alpha)  ,np.array([atom_sampling(T_steps, data, n, alpha, loss_fun, mn_0) for i in range(0, N_mc)])
    elif approx_type == 'dirichlet_multinomial':
        return dirichlet_multinomial_sampling_stacked(N_mc,T_steps, n, alpha), np.array(
            [atom_sampling(T_steps, data, n, alpha, loss_fun, mn_0)for i in range(0, N_mc)])

def h_stacked(loss_fun, theta, X):
    # Compute loss_fun at single data point xi and at parameter theta.
    # For regression loss, first column of xi is y, all other columns are X
    # print(X.shape)
    # print(theta.shape)
    # 1-D Gaussian location MLE loss
    if loss_fun == 'gaussian_loc_lik':
        h = 1.e-3 * np.square(X - theta, dtype=np.float64)
    # Linear regression loss, both y and X are real
    elif loss_fun == 'squared':
        reg_term = X[:,0] - np.matmul(X[:,1:], theta).reshape(-1,)
        h = 1.e-3 * np.square(reg_term, dtype=np.float64)
    # Logistic regression loss, y is in {-1,1}, X is real
    elif loss_fun == 'logistic':
        clas_term = X[:,0] * np.matmul(X[:,1:], theta).reshape(-1,)
        h = 1.e-3 * np.log(1 + np.exp(-clas_term, dtype=np.float64))
    # Median regression loss, both y and X are real
    elif loss_fun == 'pinball_median':
        reg_term = X[:,0] - np.matmul(X[:,1:], theta).reshape(-1,)
        h = 1.e-3 * np.abs(reg_term, dtype=np.float64)
    return h

def grad_stacked(loss_fun, theta, X):
    # Compute gradient wrt theta of loss_fun at single data point xi
    # For regression loss, first column of xi is y, all other columns are X
    # 1-D Gaussian location MLE loss
    if loss_fun == 'gaussian_loc_lik':
        g = -2 * 1.e-3 * (X - theta)
    # Regression losses, both y and X are real
    elif loss_fun == 'squared':
        reg_term = X[:, 0] - np.matmul(X[:, 1:], theta)
        g = 1.e-3 * (-2 * (reg_term.reshape(-1, 1)) * X[:, 1:])
    # Logistic regression loss, y is in {-1,1}, X is real
    elif loss_fun == 'logistic':
        clas_term = X[:,0] * np.matmul(X[:,1:], theta).reshape(-1,)
        exp_term = np.exp(-clas_term, dtype=np.float64)
        g = -1.e-3 * (X[:,0] * exp_term / (1 + exp_term)).reshape(-1,1) * X[:,1:]
    # Median regression loss, both y and X are real
    elif loss_fun == 'pinball_median':
        reg_term = X[:, 0] - np.matmul(X[:, 1:], theta)
        g = 1.e-3 * (-np.sign(reg_term.reshape(-1, 1)) * X[:, 1:])
    return g

def phi(t, beta):
    # Compute second-order utility phi at t and ambiguity index beta
    clip = min(t / beta, 700)  ## avoid overflow issues with np.exp
    return (beta * np.exp(clip)) - beta

def phi_inv(t, beta):
    # Compute inverse of second-order utility phi at t and ambiguity index beta
    return beta * np.log((t / beta) + 1)

def phi_prime(t, beta):
    # Compute derivative of second-order utility phi at t and ambiguity index beta
    clip = min(t / beta, 700)  ## avoid overflow issues with np.exp
    return np.exp(clip)

def criterion_value_stacked(loss_fun, theta, beta, criterion):
    # Compute criterion value, applying inverse transformation phi_inv
    # to make criterion values comparable across beta values
    N_mc = len(criterion)
    weights, atoms = a[0], a[1]
    return phi_inv(np.array([
        phi((h_stacked(loss_fun, theta, atoms[i]) * weights[i]).sum(), beta)
        for i in range(0, N_mc)
    ]).mean(), beta)

def SGD_alternative_stacked(loss_fun, theta_0, beta, criterion, n_passes, step_size0):
    # Perform SGD updates based on whole MC samples, so to reduce
    # computation at each iteration (the loss function is evaluated only at one sample).
    # Each MC sample is used at each pass (sampled without replacement), and the
    # procedure is repeated n_passes times
    weights_full,atoms_full=a[0],a[1]
    N = atoms_full.shape[0]
    theta_path = [theta_0]
    values_path = [criterion_value_stacked(loss_fun, theta_0, beta, criterion)]
    iteration = 1
    for t in tqdm.tqdm(range(1, n_passes + 1)):
        indexes = [idx for idx in range(0, N)]
        for n in range(1, N + 1):
            idx = np.random.choice(indexes)
            indexes = [i for i in indexes if i != idx]
            theta_tm1 = theta_path[-1]
            eta = step_size0 / (100 + np.sqrt(iteration))
            atoms = atoms_full[idx]
            weights = weights_full[idx]
            loss_vals = h_stacked(loss_fun, theta_tm1, atoms)
            grad_vals = grad_stacked(loss_fun, theta_tm1, atoms)
            theta_t = theta_tm1 - eta * phi_prime(loss_vals.dot(weights), beta) * np.matmul(weights, grad_vals)
            theta_path.append(theta_t)
            values_path.append(criterion_value_stacked(loss_fun, theta_t, beta, criterion))
            iteration += 1
    # plt.plot(np.arange(0, len(values_path)), values_path)
    # plt.show()
    return theta_path, values_path

def oos_performance(loss_fun, theta, tst_sample):
#     # Compute out-of-sample performance at theta on a test_sample
#     return np.array([h(loss_fun, theta, xi) for xi in tst_sample]).mean()
# def oos_performance_stacked(loss_fun, theta, tst_sample):
#     # Compute out-of-sample performance at theta on a test_sample
    return h_stacked(loss_fun, theta, tst_sample).mean()