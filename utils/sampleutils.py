import numpy as np


def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y,
                            rho_xy, hyper_params, greedy=False):
    # inputs must be floats
    if greedy:
        return mu_x.item(), mu_y.item()
    mean = [mu_x.item(), mu_y.item()]
    sigma_x *= np.sqrt(hyper_params.temperature)
    sigma_y *= np.sqrt(hyper_params.temperature)
    cov = [[sigma_x.item() * sigma_x.item(),
           rho_xy.item() * sigma_x.item() * sigma_y.item()],
           [rho_xy.item() * sigma_x.item() * sigma_y.item(),
           sigma_y.item() * sigma_y.item()]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def sample_univariate_normal(mu, sigma, hyper_params, greedy=False):
    # inputs must be floats
    if greedy:
        return mu.item()
    x = np.random.normal(mu.item(), sigma.item()*np.sqrt(hyper_params.temperature))
    return x
