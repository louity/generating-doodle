import torch


def kullback_leibler_loss(model, use_cuda, batch_size, annealing=True):
    LKL = -0.5*torch.sum(1 + model.sigma-model.mu**2-torch.exp(model.sigma))\
        / float(model.hyper_params.Nz*batch_size)
    if annealing:
        if use_cuda:
            KL_min = torch.Tensor([model.hyper_params.KL_min]).cuda().detach()
        else:
            KL_min = torch.Tensor([model.hyper_params.KL_min]).detach()
        return model.hyper_params.wKL*model.eta_step * torch.max(LKL, KL_min)
    else:
        return(LKL)


def reconstruction_loss_point(model, mask, dx, dy, p, max_len_out):
    pdf = model.bivariate_normal_pdf(dx, dy)
    LS = -torch.sum(mask*torch.log(1e-5+torch.sum(model.pi * pdf, 2)))\
            /float(max_len_out*model.hyper_params.batch_size)
    LP = -torch.sum(p*torch.log(model.q))/float(max_len_out*model.hyper_params.batch_size)
    return LS+LP


def reconstruction_loss_line(model, mask, dx, dy, r, phi, p0, max_len_out):
    pdf = model.bivariate_normal_pdf(dx, dy)
    LS = -torch.sum(mask*torch.log(1e-5 + torch.sum(model.pi * pdf, dim=2)))\
            /float(max_len_out*model.hyper_params.batch_size)

    pdf_r = model.univariate_normal_r_pdf(r)
    LS_r = -torch.sum(mask*torch.log(1e-5 + torch.sum(model.pi_r * pdf_r, dim=2)))\
            /float(max_len_out*model.hyper_params.batch_size)

    pdf_phi = model.univariate_normal_phi_pdf(phi)
    LS_phi = -torch.sum(mask * torch.log(1e-5 + torch.sum(model.pi_phi * pdf_phi, dim=2)))\
            /float(max_len_out * model.hyper_params.batch_size)

    LS += LS_r + LS_phi

    LP = -torch.sum(p0 * torch.log(model.q0.squeeze()))/float(max_len_out * model.hyper_params.batch_size)
    # TODO: here I change the output to track which loss is not computed well
    return LS+LP


def reconstruction_loss(model, param_info, max_len_out, type_param):
    if type_param not in ['point', 'line']:
        raise ValueError('type_param arg is not valid')
    if type_param == 'point':
        (mask, dx, dy, p) = param_info
        return reconstruction_loss_point(model, mask, dx, dy, p, max_len_out)
    elif type_param == 'line':
        (mask, dx, dy, r, phi, p0) = param_info
        return reconstruction_loss_line(model, mask, dx, dy, r, phi, p0,
                                        max_len_out)
