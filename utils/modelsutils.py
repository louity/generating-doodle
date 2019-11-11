import torch
import torch.nn.functional as F
'''
TODO: here that we should make a class out of each of the geometric element.
-> attribute: the parameter of the distribution and then we define
function that sample something from the distribution?
'''


class DistGeoBrick():
    '''
    This describes the distribution of the
    '''
    def __init__(self, hp, parametrization='point'):
        self.parametrization = parametrization
        self.M = hp.M

    def sample(self):
        # TODO: declare the type
        return(0)


def get_distr_param(y, len_out, hp, type_param='point'):
    '''
    Split and transform  the output y of LTSM into the associated parameters in
    the modelled distribution. This choice depends on the type of
    parametrization.
    '''
    batch_size = hp.batch_size
    # check we are happy with the format of y
    # if y.shape != (len_out, hp.batch_size, hp.output_size):
    #    raise ValueError('y is not of the right size')
    if type_param == 'point':
        # import pdb; pdb.set_trace()
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])
        params_pen = params[-1]
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, dim=2)
        dim_soft_pi = pi.transpose(0, 1).squeeze().shape.index(hp.M)
        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=dim_soft_pi).view(len_out, -1, hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        q = F.softmax(params_pen, dim=1).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q
    elif type_param == 'line':
        # TODO: is slicing efficient?
        idx = 6*hp.M
        params_center = y[:, :idx]
        params_radius = y[:, idx:idx + 3*hp.Mr]
        idx += 3*hp.Mr
        params_angle = y[:, idx:idx + 3*hp.Mphi]
        params_end = y[:, -1:]
        # to get the right dimensions
        params_center = torch.stack(torch.split(params_center, 6, 1))
        params_radius = torch.stack(torch.split(params_radius, 3, 1))
        params_angle = torch.stack(torch.split(params_angle, 3, 1))
        params_end = torch.stack(torch.split(params_end, 1, 1))
        # TODO: check the size of params_end, should be of size len_out ?
        # even more fine grained
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_center, 1, dim=2)
        pi_r, mu_r, sigma_r = torch.split(params_radius, 1, dim=2)
        pi_phi, mu_phi, sigma_phi = torch.split(params_angle, 1, dim=2)

        # preprocess params::
        '''
        batch_size = hp.batch_size  # TODO: replace all -1 by batch_size: not clear in fact
        OK we don't want to replace -1 by batch_size, because it is an
        argument that is not given and hence when the batch_size equals 1 we are in the cheat.
        '''
        dim_soft_pi = pi.transpose(0, 1).squeeze().shape.index(hp.M)
        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=dim_soft_pi).view(len_out, -1, hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)

        dim_soft_pi_r = pi.transpose(0, 1).squeeze().shape.index(hp.Mr)
        pi_r = F.softmax(pi_r.transpose(0, 1).squeeze(),
                         dim=dim_soft_pi_r).view(len_out, -1, hp.Mr)
        sigma_r = torch.exp(sigma_r.transpose(0, 1).squeeze()).view(len_out, -1, hp.Mr)
        # TODO: mu_r could be exp(..) to ensure that it is strictly positive...
        mu_r = mu_r.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.Mr)

        dim_soft_phi = pi_phi.transpose(0, 1).squeeze().shape.index(hp.Mphi)
        pi_phi = F.softmax(pi_phi.transpose(0, 1).squeeze(),
                           dim=dim_soft_phi).view(len_out, -1, hp.Mphi)
        sigma_phi = torch.exp(sigma_phi.transpose(0, 1).squeeze()).view(len_out, -1, hp.Mphi)
        mu_phi = mu_phi.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.Mphi)

        q = F.softmax(params_end, dim=1).view(len_out, -1, 1)

        coef_center = (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        coef_rad = (pi_r, mu_r, sigma_r)
        coef_ang = (pi_phi, mu_phi, sigma_phi)
        return (coef_center, coef_rad, coef_ang, q)
