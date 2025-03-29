import numpy as np
import plotly.graph_objects as go
import scipy.stats as sps
from scipy.integrate import quad
from sympy import symbols, exp
from scipy.stats import gaussian_kde

alpha = 1.5
lambda_ = 1
lambda_expon = 1
mu, sigma = 0, 1
a, b = 0, 1

import numpy as np
import scipy.stats as sps


def generation_dataset(N, data="constant", alpha=2.5, lambda_=1, lambda_expon=1, a=0, b=1, mu=0, sigma=1, cauchy_loc=0, cauchy_scale=1, pareto_shape=2, gpd_shape=0.5, gpd_scale=1, levy_loc=0, levy_scale=1):
    """
    Generate a dataset X of N points according to the specified distribution.
    """
    if data == "constant":
        C = np.exp(alpha * np.log(lambda_))
        Y = np.random.uniform(0, 1, N)
        X = (C / (1 - Y)) ** (1 / alpha)
    elif data == "exponential":
        X = np.random.exponential(1 / lambda_expon, N)
    elif data == "uniform":
        X = np.random.uniform(a, b, N)
    elif data == "normal":
        X = np.abs(np.random.normal(mu, sigma, N))
    elif data == "pareto":
        X = np.random.pareto(pareto_shape, N)
    elif data == "cauchy":
        X = np.random.standard_cauchy(N) * cauchy_scale + cauchy_loc
    elif data == "gpd":
        X = sps.genpareto.rvs(c=gpd_shape, scale=gpd_scale, size=N)
    elif data == "levy":
        X = sps.levy.rvs(loc=levy_loc, scale=levy_scale, size=N)
    else:
        raise ValueError("data must be one of: constant, exponential, uniform, normal, pareto, cauchy, gpd, levy")
    return X


def q(beta, data="constant", alpha=2.5, lambda_=1, lambda_expon=1, a=0, b=1, mu=0, sigma=1, 
      cauchy_loc=0, cauchy_scale=1, pareto_shape=2, gpd_shape=0.5, gpd_scale=1, levy_loc=0, levy_scale=1):
    """
    Calculate the true quantile of order beta for the specified distribution.
    """
    if data == "constant":
        C = np.exp(alpha * np.log(lambda_))
        return (C / (1 - beta)) ** (1 / alpha)
    elif data == "exponential":
        return np.log(1 / (1 - beta)) / lambda_expon
    elif data == "uniform":
        return a + beta * (b - a)
    elif data == "normal":
        return sps.norm.ppf((1 + beta) / 2, loc=mu, scale=sigma)
    elif data == "pareto":
        return (1 - beta) ** (-1 / pareto_shape) - 1
    elif data == "cauchy":
        return sps.cauchy.ppf(beta, loc=cauchy_loc, scale=cauchy_scale)
    elif data == "gpd":
        return sps.genpareto.ppf(beta, c=gpd_shape, scale=gpd_scale)
    elif data == "levy":
        return sps.levy.ppf(beta, loc=levy_loc, scale=levy_scale)
    else:
        raise ValueError("data must be one of: constant, exponential, uniform, normal, pareto, cauchy, gpd, levy")

def naive_hill_estimator(X, beta):
    """
    Estimateur de Hill dans le cas simple d'une de distribution en C x^{-\alpha}
    La fonction retourne q_{\beta}
    """
    X_sort = np.sort(X)
    lambda_, alpha = X_sort[0], 1 / np.mean(np.log(X_sort) - np.log(X_sort[0]))
    C = np.exp(alpha * np.log(lambda_))
    return (C / (1 - beta)) ** (1 / alpha)


def hill_estimator(X, beta):
    """
    Estimateur de Hill dans le cas général d'une de distribution en L(x) x^{-\alpha}
    """
    N = len(X)
    K_N = int(np.sqrt(N))
    X_sort = np.sort(X)
    X_sort = X_sort[-K_N:]
    lambda_, alpha = X_sort[0], 1 / np.mean(np.log(X_sort) - np.log(X_sort[0]))
    # return ((N * (1 - beta) / K_N)**(-alpha)) * lambda_
    return lambda_ * (N / K_N * (1 - beta)) ** (-1 / alpha)

def hill_density(X, x):
    N = len(X)
    K_N = int(np.sqrt(N))
    X_sort = np.sort(X)
    X_sort = X_sort[-K_N:]
    lambda_, alpha = X_sort[0], 1 / np.mean(np.log(X_sort) - np.log(X_sort[0]))
    return alpha * (K_N / N) * (x / lambda_) ** (-alpha - 1)


def test_on_synthetic_data(N, M, data="constant"):
    betas = np.linspace(0.1, 0.9, 100)

    # Simulation des estimateurs
    hill_means = []
    hill_stds = []
    hill_variable_means = []
    hill_variable_stds = []

    for beta in betas:
        hill_vals = [naive_hill_estimator(generation_dataset(N, data, alpha=2.5, lambda_=1, lambda_expon=1, a=0, b=1, mu=0, sigma=1, cauchy_loc=0, cauchy_scale=1, pareto_shape=2, gpd_shape=0.5, gpd_scale=1, levy_loc=0, levy_scale=1), beta) for _ in range(M)]
        hill_variable_vals = [hill_estimator(generation_dataset(N, data, alpha=2.5, lambda_=1, lambda_expon=1, a=0, b=1, mu=0, sigma=1, cauchy_loc=0, cauchy_scale=1, pareto_shape=2, gpd_shape=0.5, gpd_scale=1, levy_loc=0, levy_scale=1), beta) for _ in range(M)]
        
        hill_means.append(np.mean(hill_vals))
        hill_stds.append(np.std(hill_vals))
        
        hill_variable_means.append(np.mean(hill_variable_vals))
        hill_variable_stds.append(np.std(hill_variable_vals))

    # Quantile réel
    q_true = [q(beta, data, alpha=2.5, lambda_=1, lambda_expon=1, a=0, b=1, mu=0, sigma=1, 
      cauchy_loc=0, cauchy_scale=1, pareto_shape=2, gpd_shape=0.5, gpd_scale=1, levy_loc=0, levy_scale=1) for beta in betas]

    return betas, q_true, hill_means, hill_stds, hill_variable_means, hill_variable_stds, data

def test_on_real_data(X):
    betas = np.linspace(0.1, 0.9, 100)
    hill_quantiles = []

    for beta in betas:
        hill_quantiles += [hill_estimator(X, beta)]

    return betas, hill_quantiles, X

def plot_hill_estimation_real_data(betas, hill_quantiles, X):
    bandwidth_factor=2
    
    # Concatenate all values (positive and negative)
    X_combined = np.concatenate((X[X > 0], X[X < 0]))

    # Plot Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=X_combined, histnorm='probability density', name='Données réelles', opacity=0.75, marker_color='blue'))

    # Estimate density function with KDE using a smoother bandwidth
    kde = gaussian_kde(X_combined, bw_method='silverman')
    kde.set_bandwidth(bw_method=kde.factor * bandwidth_factor)  # Increase bandwidth for smoother density

    # Generate the density curve
    x_range = np.linspace(min(X_combined), max(X_combined), 1000)
    kde_values = kde(x_range)

    # Add KDE plot
    fig.add_trace(go.Scatter(x=x_range, y=kde_values, mode='lines', name='Densité estimée (KDE)', line=dict(color='red')))

    # Estimated density
    x_range_hill = np.linspace(0.5, max(X_combined), 1000)
    hill_density_values = [hill_density(X, x) for x in x_range_hill]
    hill_density_values = np.array(hill_density_values) / np.sum(hill_density_values)  # Normalize the density
    fig.add_trace(go.Scatter(x=x_range_hill, y=hill_density_values, mode='lines', name='Densité estimée (Hill)', line=dict(color='green')))

    fig.update_layout(title='Histogramme et Densité des données réelles', xaxis_title='Valeur', yaxis_title='Densité de probabilité')
    fig.show()

    # Plot Hill Estimator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=hill_quantiles, mode='lines', name='Estimateur de Hill'))
    fig.update_layout(title="Estimateur de Hill pour des données réelles", xaxis_title='Beta', yaxis_title="Valeur de l'estimateur")
    fig.show()


def plot_hill_estimation(betas, q_true, hill_means, hill_stds, hill_variable_means, hill_variable_stds, data):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=betas, y=np.array(hill_means) + np.array(hill_stds), fill=None, mode='lines', line=dict(color='rgba(0,0,255,0.1)'), showlegend=False))
    fig.add_trace(go.Scatter(x=betas, y=np.array(hill_means) - np.array(hill_stds), fill='tonexty', mode='lines', line=dict(color='rgba(0,0,255,0.1)'), showlegend=False))
    fig.add_trace(go.Scatter(x=betas, y=hill_means, mode='lines', name='Estimateur de Hill naïf', line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=betas, y=np.array(hill_variable_means) + np.array(hill_variable_stds), fill=None, mode='lines', line=dict(color='rgba(255,0,0,0.1)'), showlegend=False))
    fig.add_trace(go.Scatter(x=betas, y=np.array(hill_variable_means) - np.array(hill_variable_stds), fill='tonexty', mode='lines', line=dict(color='rgba(255,0,0,0.1)'), showlegend=False))
    fig.add_trace(go.Scatter(x=betas, y=hill_variable_means, mode='lines', name='Estimateur de Hill', line=dict(color='red')))

    fig.add_trace(go.Scatter(x=betas, y=q_true, mode='lines', name='Quantile réel', line=dict(color='green')))

    fig.update_layout(title=f'Estimateurs de Hill pour des données de type {data}', xaxis_title='Beta', yaxis_title='Valeur de l\'estimateur')
    fig.show()

def plot_error_estimation(betas, q_true, hill_means, hill_stds, hill_variable_means, hill_variable_stds, data):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=betas, y=100 * np.abs(np.array(hill_means) - q_true) / np.abs(q_true), mode='lines', name='Hill constant', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=betas, y=100 * np.abs(np.array(hill_variable_means) - q_true) / np.abs(q_true), mode='lines', name='Hill variable', line=dict(color='red')))

    fig.update_layout(title=f'Estimateurs de Hill pour des données de type {data}', xaxis_title='Beta', yaxis_title='Erreur de l\'estimateur en %')
    fig.show()