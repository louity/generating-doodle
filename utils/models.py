import torch
from torch import nn

from utils.modelsutils import get_distr_param


class EncoderRNN(nn.Module):
    def __init__(self, hp):
        super(EncoderRNN, self).__init__()
        if not hasattr(hp, 'use_cuda'):
            raise ValueError('hp should have an attribute use_cuda')
        self.hyper_params = hp
        # bidirectional lstm:
        # TODO: change LSTM(hp.size_paramatrization)
        self.lstm = nn.LSTM(self.hyper_params.size_paramatrization,
                            self.hyper_params.enc_hidden_size,
                            bidirectional=True)
        self.dropout = nn.Dropout(self.hyper_params.dropout)
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2*self.hyper_params.enc_hidden_size,
                               hp.Nz)
        self.fc_sigma = nn.Linear(2*self.hyper_params.enc_hidden_size,
                                  self.hyper_params.Nz)
        # active dropout:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        # (self, inputs, batch_size, use_cuda=True, hidden_cell=None)
        # TODO: isn't it redundant to have inputs and batch_size???
        if hidden_cell is None:
            # then must init with zeros
            if self.hyper_params.use_cuda:
                hidden = torch.zeros(2, batch_size,
                                     self.hyper_params.enc_hidden_size).cuda()
                cell = torch.zeros(2, batch_size,
                                   self.hyper_params.enc_hidden_size).cuda()
            else:
                hidden = torch.zeros(2, batch_size,
                                     self.hyper_params.enc_hidden_size)
                cell = torch.zeros(2, batch_size,
                                   self.hyper_params.enc_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(self.dropout(hidden),
                                                      1, 0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0),
                                hidden_backward.squeeze(0)], 1)
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat/2.)
        # N ~ N(0,1)
        z_size = mu.size()
        if self.hyper_params.use_cuda:
            N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        z = mu + sigma*N
        # mu and sigma_hat are needed for LKL loss
        return z, mu, sigma_hat


class DecoderRNN(nn.Module):
    def __init__(self, hp, max_len_out=10):
        super(DecoderRNN, self).__init__()
        if not hasattr(hp, 'use_cuda'):
            raise ValueError('hp should have an attribute use_cuda')
        self.hyper_params = hp
        self.max_len_out = max_len_out
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(self.hyper_params.Nz, 2*hp.dec_hidden_size)
        # unidirectional lstm:
        # TODO: d'ou vient le + 5??
        # self.lstm = nn.LSTM(self.hyper_params.Nz + 5, hp.dec_hidden_size)
        self.lstm = nn.LSTM(self.hyper_params.Nz +
                            self.hyper_params.size_parametrization,
                            hp.dec_hidden_size)
        self.dropout = nn.Dropout(self.hyper_params.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(self.hyper_params.dec_hidden_size,
                                   6*self.hyper_params.M+3)
        # TODO: why 6*self.hyper_params.M + 3 ?
        self.hyper_params.output_size = 6*self.hyper_params.M + 3

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)),
                                       self.hyper_params.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(),
                           cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1,
                               self.hyper_params.dec_hidden_size))
            len_out = self.max_len_out + 1
        else:
            y = self.fc_params(self.dropout(hidden).view(-1,
                               self.hyper_params.dec_hidden_size))
            len_out = 1
        res = get_distr_param(y, len_out,
                              self.hyper_params, type_param='point')
        return (*res, hidden, cell)

'''
class DecoderRNN_alone(nn.Module):
    def __init__(self, hp, max_len_out=10):
        super(DecoderRNN, self).__init__()
        if not hasattr(hp, 'use_cuda'):
            raise ValueError('hp should have an attribute use_cuda')
        self.hp = hp
        self.max_len_out = max_len_out
        # to init hidden and cell from z:
        # self.fc_hc = nn.Linear(hp.Nz, 2*hp.dec_hidden_size)
        # TODO: hc stands for hidden-cell? But fc what does it mean?
        self.fc_hc = nn.Linear(in_feature=hp.Nz,
                               out_feature=2*hp.dec_hidden_size)
        # unidirectional lstm:
        # self.lstm = nn.LSTM(hp.Nz+5, hp.dec_hidden_size)
        self.lstm = nn.LSTM(input_size=5,
                            hidden_size=hp.dec_hidden_size)
        self.dropout = nn.Dropout(hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6*hp.M+3)
        self.hp.output_size = 6*hp.M + 3

    def forward(self, inputs, hidden_cell=None):
        if hidden_cell is not None:
            outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        else:
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)),
                                       self.hp.dec_hidden_size, 1)
            outputs, (hidden, cell) = self.lstm(inputs)

        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
            len_out = self.max_len_out + 1
        else:
            y = self.fc_params(self.dropout(hidden).view(-1,
                               self.hp.dec_hidden_size))
            len_out = 1
        res = get_distr_param(y, len_out, self.hp, type_param='point')
        return (*res, hidden, cell)
'''
