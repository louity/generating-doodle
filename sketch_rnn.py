import argparse

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
from torch import optim, nn

from utils.batchutils import make_batch_point, make_batch_line
from utils.batchutils import make_target_point, make_target_line
from utils.loss import reconstruction_loss, kullback_leibler_loss
from utils.models import EncoderRNN, DecoderRNN
# from utils.models_line import EncoderRNN_line, DecoderRNN_line
from utils.sampleutils import sample_bivariate_normal, sample_univariate_normal
from utils.visutils import make_image, make_image_seq, plot_sketch
from utils.dataset.datautils import purify, normalize_strokes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch-RNN training Training')
    parser.add_argument('--parametrization', default='point',
                        choices=['point', 'line'],
                        help='Parametrization of the drawing (line or point)')
    parser.add_argument('--train_data', default='data/broccoli_car_cat.npz',
                        help='Numpy zip containing the train data')
    parser.add_argument('--plot_random_train_data', action='store_true',
                        help='plot a randomly chosen train data')
    parser.add_argument('--experiment', default='uncondition',
                        choices=['uncondition', 'complete'],
                        help='try uncondition of to complete generation')
    parser.add_argument('--sigma', default=1,
                        help='variance when generating a point')

    args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# parametrization = args.parametrization


class HParams():
    def __init__(self):
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.1
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200
        #if parametrization == 'line':
        #    self.Mr = 10  # nbr mixture gaussian for radius
        #    self.Mphi = 10  # nbr mixture gaussian for angle


# hp = HParams()
# hp.use_cuda = use_cuda


class DataLoader():
    # TODO: Only doing it for point
    def __init__(self, path_data, hp):
        '''
        loc_train_data : the path to the data
        hp : an instance of HParams
        '''
        # dataset = np.load(args.train_data, encoding='latin1')
        dataset = np.load(path_data, encoding='latin1')
        self.path_data = path_data
        self.data = dataset['train']
        self.valid_set = dataset['valid']
        self.test_set = dataset['test']  # TODO: unused for now
        # preprocess the data
        self.data = purify(self.data, hp)
        self.data = normalize_strokes(self.data)
        self.max_len_out = max([len(seq) for seq in self.data])
        # preprocessing the valid data
        self.valid_set = purify(self.valid_set, hp)
        self.valid_set = normalize_strokes(self.valid_set)
        self.max_len_out_val = max([len(seq) for seq in self.valid_set])

    def make_batch(self, nbr_datum,
                   use_cuda,
                   parametrization='point',
                   type_set='train'):
        # TODO: adapt make_batch to another parametrization
        # randomly selecting nbr_datum datum among the data set
        if type_set not in ['train', 'valid', 'test']:
            raise ValueError('this type of dataset does not exist')
        if type_set == 'train':
            l_idx = np.random.choice(len(self.data), nbr_datum, replace='False')
            batch_sequences = [self.data[i] for i in l_idx]
        elif type == 'valid':
            l_idx = np.random.choice(len(self.valid_set), nbr_datum)
            batch_sequences = [self.valid_set[i] for i in l_idx]
        else:
            raise ValueError('no use for now of test')
        return(make_batch_point(batch_sequences,
                                self.max_len_out,
                                use_cuda))

    # TODO: something that read directly an input Image and process
    # it for the network.

    def select_batch(self,
                     l_idx_datum,
                     use_cuda,
                     type_set='train'):
        # selecting specific idx
        # check that idx are valid
        if not all(i >= 0 for i in l_idx_datum):
            raise ValueError('some index in l_idx_datum is negative')
        if type_set == 'train':
            if not all(i < len(self.data) for i in l_idx_datum):
                raise ValueError('some index in l_idx_datum is to big')
            batch_sequences = [self.data[i] for i in l_idx_datum]
        elif type_set == 'valid':
            if not all(i < len(self.valid_set) for i in l_idx_datum):
                raise ValueError('some index in l_idx_datum is to big')
            batch_sequences = [self.data[i] for i in l_idx_datum]
        else:
            raise ValueError('for now no use of test')
        # TODO: change self.max_len_out
        return(make_batch_point(batch_sequences,
                                self.max_len_out,
                                use_cuda))

    def plot_image(self, idx, plot=True):
        off_seq = self.data[idx]

        def make_seq(seq_x, seq_y, seq_z):
            # transform the lists in the right array
            x_sample = np.cumsum(seq_x, 0)
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
            sequence_coo = np.stack([x_sample, y_sample, z_sample]).T
            sequence_offset = np.stack([np.array(seq_x),
                                       np.array(seq_y), np.array(z_sample)]).T
            return(sequence_coo, sequence_offset)
        seq, _ = make_seq(off_seq[:, 0], off_seq[:, 1], off_seq[:, 2])
        # TODO: still not sure of what is off_seq
        make_image(seq, dest_folder=None,
                   name='_output_', plot=plot)

    def initialize(self,
                   parametrization,
                   use_cuda,
                   batch_size):
        if parametrization == 'point':
            # create start of sequence:
            if use_cuda:
                sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] *
                                  batch_size).cuda().unsqueeze(0)
            else:
                sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] *
                                  batch_size).unsqueeze(0)
        elif parametrization == 'line':
            if use_cuda:
                sos = torch.stack([torch.Tensor([0, 0, 0, 0, 0])] *
                                  batch_size).cuda().unsqueeze(0)
            else:
                sos = torch.stack([torch.Tensor([0, 0, 0, 0, 0])] *
                                  batch_size).unsqueeze(0)
        return(sos)


def lr_decay(optimizer, hp):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


class Model():
    def __init__(self, hyper_parameters, parametrization='point'):

        self.parametrization = parametrization
        self.hyper_params = hyper_parameters
        if self.parametrization == 'point':
            self.hyper_params.size_paramatrization = 5
            self.encoder = EncoderRNN(self.hyper_params)
            self.decoder = DecoderRNN(self.hyper_params,
                                      max_len_out=self.hyper_params.max_len_out)
        elif self.parametrization == 'line':
            # import pdb; pdb.set_trace()
            self.encoder = EncoderRNN_line(self.hyper_params)
            self.decoder = DecoderRNN_line(self.hyper_params, max_len_out=self.hyper_params.max_len_out)

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), self.hyper_params.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), self.hyper_params.lr)
        self.eta_step = self.hyper_params.eta_min
        # keep track of loss value
        self.loss_train = []
        self.loss_valid = []

    def train(self, epoch, dataloader):
        '''
        #TODO : fill
        '''
        if not hasattr(self, 'dataloader'):
            self.dataloader = dataloader
        self.encoder.train()
        self.decoder.train()

        # prepare batch
        batch, lengths = dataloader.make_batch(self.hyper_params.batch_size,
                                               use_cuda,
                                               parametrization=self.parametrization)
        # encode:
        z, self.mu, self.sigma = self.encoder(batch,
                                              self.hyper_params.batch_size)
        # TODO: replace by 'point' by self.parametrisation
        # self.parametrization devrait etre dans dataloader???
        sos = dataloader.initialize('point',
                                    use_cuda,
                                    self.hyper_params.batch_size)
        batch_init = torch.cat([sos, batch], 0)
        z_stack = torch.stack([z]*(self.hyper_params.max_len_out+1))
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack], 2)
        # decode:
        if self.parametrization == 'point':
            # import pdb; pdb.set_trace()
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                    self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        elif self.parametrization == 'line':
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                    self.rho_xy, self.pi_r, self.mu_r, self.sigma_r, self.pi_phi, self.mu_phi \
                    , self.sigma_phi, self.q0, _, _ = self.decoder(inputs, z)

        # prepare targets:
        # TODO: create fct make_target that takes parametrization as variable
        if self.parametrization == 'point':
            mask, dx, dy, p = make_target_point(batch, lengths, self.hyper_params,
                                                self.hyper_params.max_len_out, use_cuda)
        elif self.parametrization == 'line':
            # TODO is it batch of batch_ini that we xant to give?
            mask, dx, dy, r, phi, p0 = make_target_line(batch, lengths,
                                                        self.hyper_params,
                                                        self.hyper_params.max_len_out, use_cuda)

        if dx.shape[:2] != self.mu_x.shape[:2]:
            print(dx.shape[:2])
            print(self.mu_x.shape[:2])
            raise ValueError('batch et target batch output tensor not having\
                             same shape')
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1-(1-self.hyper_params.eta_min)*self.hyper_params.R
        # compute losses:
        LKL = kullback_leibler_loss(self, use_cuda,
                                    self.hyper_params.batch_size)
        if self.parametrization == 'point':
            param_info = (mask, dx, dy, p)
        elif self.parametrization == 'line':
            param_info = (mask, dx, dy, r, phi, p0)
        LR = reconstruction_loss(self, param_info, self.hyper_params.max_len_out,
                                 self.parametrization)
        loss = LR + LKL
        self.size_checkpoint = 1000
        # TODO: add checkpoint as an argument

        # gradient step
        loss.backward()
        # gradient cliping
        nn.utils.clip_grad_norm_(self.encoder.parameters(),
                                 self.hyper_params.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(),
                                 self.hyper_params.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if epoch % 100 == 0:
            # before the lr_decay was every epoch
            print('epoch', epoch, 'loss', loss.item(), 'LR',
                  LR.item(), 'LKL', LKL.item())
            self.encoder_optimizer = lr_decay(self.encoder_optimizer,
                                              self.hyper_params)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer,
                                              self.hyper_params)
        if epoch % self.size_checkpoint == 0 and epoch != 0:
            self.save(epoch)
            # Beware here I added the .item()
            self.loss_train.append(loss.item())
            self.loss_valid.append(self.compute_loss_valid().item())
            if self.parametrization == 'point':
                self.conditional_generation_point()
            elif self.parametrization == 'line':
                self.conditional_generation_line()

    def compute_loss_valid(self) -> None:
        '''compute loss of validation set'''
        # TODO: what should we do of valid_set? What about max_len_out
        # batch, lengths = make_batch_point(valid_set,
        #                                 max_len_out_val, use_cuda)
        valid_set = self.dataloader.valid_set
        max_len_out_val = self.dataloader.max_len_out_val
        batch, lengths = make_batch_point(valid_set,
                                          self.dataloader.max_len_out_val,
                                          use_cuda)
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)

        # encode:
        z, self.mu, self.sigma = self.encoder(batch, len(valid_set))

        # TODO: print the sum of self.mu and self.sigma on the

        # create start of sequence:
        if use_cuda:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] *
                              len(valid_set)).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] *
                              len(valid_set)).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch], 0)
        # expend z to be ready to concatenate with inputs
        z_stack = torch.stack([z]*(max_len_out_val + 1))
        # TODO: to complete from here...
        inputs = torch.cat([batch_init, z_stack], 2)
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        mask, dx, dy, p = make_target_point(batch, lengths, self.hyper_params,
                                            max_len_out_val, use_cuda)
        LKL = kullback_leibler_loss(self, use_cuda, len(valid_set),
                                    annealing=False)
        param_info = (mask, dx, dy, p)
        LR = reconstruction_loss(self, param_info, self.hyper_params.max_len_out,
                                 self.parametrization)
        loss = LR + LKL
        return loss

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp/norm

    def univariate_normal_r_pdf(self, r):
        exp = torch.exp(-(r-self.mu_r)**2/(2*self.sigma_r**2))/(torch.sqrt(
            torch.tensor(2*np.pi))*self.sigma_r)
        return exp

    def univariate_normal_phi_pdf(self, phi):
        exp = torch.exp(-(phi-self.mu_phi)**2/(2*self.sigma_phi**2))/(
            torch.sqrt(torch.tensor(2*np.pi))*self.sigma_phi)
        return exp

    def save(self, epoch):
        #if args.train_data[:4] != 'data':
        if self.dataloader.path_data[:4] != 'data':
            raise ValueError('the train data should be in folder data')
        #if args.train_data[-4:] != '.npz':
        if self.dataloader.path_data[-4:] != '.npz':
            raise ValueError('the train data should be of extension .npz')
        self.type_drawing = args.train_data[5:-4]
        torch.save(self.encoder.state_dict(),
                   'save/encoder_' + self.type_drawing + '_{}.pth'.format(epoch))
        #           'save/' + self.parametrization +
        #           '_encoderRNN_epoch_%d.pth' % (epoch))
        torch.save(self.decoder.state_dict(),
                   'save/decoder_' + self.type_drawing + '_{}.pth'.format(epoch))
        #           'save/' + self.parametrization +
        #           '_decoderRNN_epoch_%d.pth' % (epoch))

    def load(self, encoder_name, decoder_name, use_cuda=use_cuda):
        if self.hyper_params.use_cuda:
            saved_encoder = torch.load(encoder_name)
            saved_decoder = torch.load(decoder_name)
        else:
            saved_encoder = torch.load(encoder_name, map_location='cpu')
            saved_decoder = torch.load(decoder_name, map_location='cpu')
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation_point(self,
                                     uncondition=False,
                                     plot=False,
                                     sigma=1,
                                     idx_image=11):
        '''
        uncondition : If True, then it decodes an image starting from
        a random image. If False, it select a random image, encode it and then
        decode a new image with the latent vector coming from it.

        Input:
        ------
            - nbr_image : index of the image which condition the generation
            if uncondition is False.
            - sigma : variance of the Gaussian Vector if generating from a
            random gaussian vector (if uncondition is True)
        '''
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)
        # encode:
        if uncondition:
            # z is just a random normal of size Nz?
            # TODO: I changed this hp.Nz
            if use_cuda:
                z = torch.from_numpy(np.random.normal(0, sigma, self.hyper_params.Nz))\
                    .type('torch.FloatTensor').view(-1, self.hyper_params.Nz).cuda()
            else:
                z = torch.from_numpy(np.random.normal(0, sigma, self.hyper_params.Nz))\
                    .type('torch.FloatTensor').view(-1, self.hyper_params.Nz)
        else:
            if not hasattr(self, 'dataloader'):
                raise ValueError('To have the latent image of an image, you need \
                                 the dataloader for that image.')
            batch, lengths = self.dataloader.select_batch([idx_image],
                                                          use_cuda)
            z, _, _ = self.encoder(batch, 1)

        if use_cuda:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
        else:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(self.hyper_params.max_len_out):
            if use_cuda:
                input = torch.cat([s, z.unsqueeze(0).cuda()], 2)
            else:
                input = torch.cat([s, z.unsqueeze(0)], 2)
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state_point()
            # ------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                print(i)
                break
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        make_image(sequence,
                   dest_folder='images/train_point',
                   plot=plot)

    def finish_drawing_point(self,
                             img_to_complete,
                             use_cuda_loc,
                             nbr_point_next=10,
                             sigma=0.1,
                             img_full=None):
        '''
        Input:
        ------
        nbr_point : number of points of the next draw
        img_to_complete : explicit
        use_cuda_loc : True/False weither or not to use cuda
        sigma : the variance of the latent vector
        img_full : if not None then pass the img through the encoder to get a
        latent vector z otherwise use a random vector z.

        img is a set of points of lengths. (dx, dy, p)
        Beware img should already be the offsets points.
        eta should be a fraction of max_len_out.

        Mais la data est initialement de la forme (dx, dy, p), pas encore de la forme
        (dx, dy, p1, p2, p3)?
        '''
        if img_full is not None:
            (nbr_point_full, _) = img_full.shape
        (nbr_point_comp, _) = img_to_complete.shape

        def make_seq(seq_x, seq_y, seq_z):
            # transform the lists in the right array
            x_sample = np.cumsum(seq_x, 0)
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
            sequence_coo = np.stack([x_sample, y_sample, z_sample]).T
            sequence_offset = np.stack([np.array(seq_x),
                                       np.array(seq_y), np.array(z_sample)]).T
            return(sequence_coo, sequence_offset)

        if use_cuda_loc:
            # TODO: should we do a unsqueeze as before?
            if img_full is not None:
                img_tensor_full = torch.tensor(img_full).view(nbr_point_full, 1, -1).cuda()
            img_tensor_comp = torch.tensor(img_to_complete).view(nbr_point_comp, 1, -1).cuda()
        else:
            if img_full is not None:
                img_tensor_full = torch.tensor(img_full).view(nbr_point_full, 1, -1)
            img_tensor_comp = torch.tensor(img_to_complete).view(nbr_point_comp, 1, -1)

        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)

        # Give a vector latent for decoding
        if img_full is not None:
            # If willing to look like a given image
            # TODO: put it on cuda like array
            z, _, _ = self.encoder(img_tensor_full, 1)
        else:
            if use_cuda_loc:
                # TODO: is it really choising the variance parameter???
                z = torch.from_numpy(np.random.normal(0, sigma, self.hyper_params.Nz))\
                    .type('torch.FloatTensor').view(-1, self.hyper_params.Nz).cuda()
            else:
                z = torch.from_numpy(np.random.normal(0, sigma, self.hyper_params.Nz))\
                    .type('torch.FloatTensor').view(-1, self.hyper_params.Nz)

        # Find the hidden state of the incomplete drawing?
        # TODO: should we also find the cell state?
        hidden_cell = None
        for i in range(nbr_point_comp):
            if use_cuda_loc:
                s = img_tensor_comp[i, :, :].type('torch.FloatTensor').view(1, 1, -1).cuda()
            else:
                s = img_tensor_comp[i, :, :].type('torch.FloatTensor').view(1, 1, -1)
            # TODO: check it is the right dimension. SHould we add also cuda?
            input = torch.cat([s, z.unsqueeze(0)], dim=2)
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)

        # complete the incomplete drawing.
        seq_x = []
        seq_y = []
        seq_z = []
        # before it was int(eta * self.hyper_params.max_len_out)
        for i in range(min(nbr_point_next, self.hyper_params.max_len_out)):
            # TODO: Is s the last point of the image?
            input = torch.cat([s, z.unsqueeze(0)], 2)
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            # TODO: there should be a temperature parameter?
            s, dx, dy, pen_down, eos = self.sample_next_state_point()
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos and i > 30:
                (seq_co, seq_of) = make_seq(seq_x, seq_y, seq_z)
                return(seq_of)
        # output result!
        (seq_co, seq_of) = make_seq(seq_x, seq_y, seq_z)
        return(seq_of)

    def sample_next_state_point(self):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/self.hyper_params.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hyper_params.M, p=pi)
        # get pen state:
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                       self.hyper_params, greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        if use_cuda:
            return next_state.cuda().view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
        else:
            return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2

    def sample_next_state_line(self):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/self.hyper_params.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture weights
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(self.hyper_params.M, p=pi)

        pi_r = self.pi_r.data[0, 0, :].cpu().numpy()
        pi_r = adjust_temp(pi_r)
        pi_r_idx = np.random.choice(self.hyper_params.Mr, p=pi_r)

        pi_phi = self.pi_phi.data[0, 0, :].cpu().numpy()
        pi_phi = adjust_temp(pi_phi)
        pi_phi_idx = np.random.choice(self.hyper_params.Mphi, p=pi_phi)

        # get pen state:
        q0 = self.q0.data[0, 0, :].cpu().numpy()
        q0 = adjust_temp(q0)
        # TODO: check if q0 is of dimension 2
        p0 = np.random.choice(2, p=[q0.item(), 1 - q0.item()])

        # get mixture params:
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, self.hyper_params, greedy=False)

        mu_r = self.mu_r.data[0, 0, pi_r_idx]
        sigma_r = self.sigma_r.data[0, 0, pi_r_idx]
        r = sample_univariate_normal(mu_r, sigma_r, self.hyper_params, greedy=False)

        mu_phi = self.mu_phi.data[0, 0, pi_phi_idx]
        sigma_phi = self.sigma_phi.data[0, 0, pi_phi_idx]
        # TODO: should think if we really want to take value modulo 2 pi
        phi = sample_univariate_normal(mu_phi, sigma_phi, self.hyper_params, greedy=False) % 2*np.pi

        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[2] = r
        next_state[3] = phi
        next_state[4] = p0
        if use_cuda:
            return next_state.cuda().view(1, 1, -1), x, y, r, phi, p0 == 1
        else:
            return next_state.view(1, 1, -1), x, y, r, phi, p0 == 1


if __name__ == "__main__":
    # beware use_cuda is not set in the hyper_parameters
    print('use_cuda is {}'.format(use_cuda))
    hp = HParams()
    hp.use_cuda = use_cuda
    # model = Model(hyper_parameters=hp, parametrization=args.parametrization)
    dataloader = DataLoader(args.train_data, hp)
    hp.max_len_out = dataloader.max_len_out
    model = Model(hyper_parameters=hp, parametrization=args.parametrization)
    for epoch in range((20001)):
        model.train(epoch, dataloader)
    # save the model loss as pickles?
    loss_train = model.loss_train
    loss_valid = model.loss_valid

    plt.figure(1)
    plt.plot(np.arange(len(loss_train)), np.array(loss_train))
    plt.plot(np.arange(len(loss_valid)), np.array(loss_valid))
    plt.legend(['train loss', 'validation loss'])
    plt.show()

    # export the model that gives the lowest on the validation set.
    idx_small = np.argmin(np.array(loss_valid))
    name_best_model = model.type_drawing + '_{}.pth'.format(int(idx_small * model.size_checkpoint))
    # save the best model in the directory specific from drawing
    os.rename("save/encoder_" + name_best_model, "draw_models/encoder_" + name_best_model)
    os.rename("save/decoder_" + name_best_model, "draw_models/decoder_" + name_best_model)

    # save the last one
    name_last_model = model.type_drawing + '_{}.pth'.format(int(len(loss_valid) * model.size_checkpoint))
    os.rename("save/encoder_" + name_last_model, "draw_models/encoder_" + name_last_model)
    os.rename("save/decoder_" + name_last_model, "draw_models/decoder_" + name_last_model)

    # save the hp parameter in draw_models/hp_folder
    name_file_hp = model.type_drawing + '_{}.pickle'.format(int(len(loss_valid) * model.size_checkpoint))
    with open('draw_models/hp_folder/' + name_file_hp, 'wb') as f:
        pickle.dump(hp, f, protocol=pickle.HIGHEST_PROTOCOL)
