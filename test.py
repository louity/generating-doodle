'''
File testing all functionalities
'''
import pickle
import torch

from sketch_rnn import Model, DataLoader
from sketch_rnn import HParams


def try_training(use_cuda):
    '''
    '''
    hp = HParams()
    hp.use_cuda = use_cuda
    dataloader = DataLoader('data/cat.npz', hp)
    hp.max_len_out = dataloader.max_len_out
    model = Model(hyper_parameters=hp, parametrization='point')

    for epoch in range((3)):
        model.train(epoch, dataloader)

    print('taining with cuda = {} works'.format(use_cuda))


def try_generating(use_cuda, sigma,
                   uncondition=False,
                   nbr_image=10):
    '''
    uncondition is True: generate a cat from a random vector
    uncondition is False: generate a cat from an initial image of cat
    '''
    hp_path = 'draw_models/hp_folder/cat_20000.pickle'
    with open(hp_path, 'rb') as handle:
        hp = pickle.load(handle)
    hp.use_cuda = use_cuda
    # load model
    model = Model(hyper_parameters=hp, parametrization='point')
    encoder_name = 'draw_models/encoder_cat_20000.pth'
    decoder_name = 'draw_models/decoder_cat_20000.pth'

    model.load(encoder_name, decoder_name)
    print('printing image we will regenerate')
    # TODO: print image corresponding to 10
    print('printing image re-generated')

    model.conditional_generation_point(10,
                                       uncondition=True,
                                       plot=True,
                                       sigma=sigma)
    print('done generating from initial image with sigma = {}'.format(sigma))


def try_completing(use_cuda, sigma):
    return()


if __name__ == '__main__':
    # try_training(False)
    try_generating(False, 0.01)
    if torch.cuda.is_available():
        try_training(True)
        try_completing(True, 0.01)
    else:
        print('could not try with cuda because not available')
    # check that there exists cuda
    # try_training(True)
