'''
File quickly testing the functionalities on a particular model.
'''
import torch

from sketch_rnn import Model, DataLoader
from sketch_rnn import HParams

if torch.cuda.is_available() == True:
    device = 'cuda'
    use_cuda = True
else:
    device = 'cpu'
    use_cuda = False

def try_training(use_cuda):
    hp = HParams()
    hp.use_cuda = use_cuda
    dataloader = DataLoader('data/cat.npz', hp)
    hp.max_len_out = dataloader.max_len_out
    model = Model(hyper_parameters=hp, parametrization='point')

    for epoch in range((3)):
        model.train(epoch, dataloader)

    print('taining with cuda = {} works'.format(use_cuda))


def try_generating(sigma,
                   encoder_name,
                   decoder_name,
                   uncondition=False,
                   idx_image=10):
    '''
    uncondition is True: generate a cat from a random vector
    uncondition is False: generate a cat from an initial image of cat
    '''
    # hp_path = 'draw_models/hp_folder/cat_20000.pickle'
    # with open(hp_path, 'rb') as handle:
    #    hp = pickle.load(handle)
    hp = HParams()
    hp.use_cuda = use_cuda
    hp.max_len_out = 300 # TODO: to change!!
    # dataLoader associated to the data set
    dataloader = DataLoader('data/cat.npz', hp)
    # load model
    model = Model(hyper_parameters=hp, parametrization='point')
    model.load(encoder_name, decoder_name)
    if not uncondition:
        print('printing image we will regenerate')
        dataloader.plot_image(idx_image, plot=True)
    print('printing generated image')
    model.dataloader = dataloader
    model.conditional_generation_point(uncondition=uncondition,
                                       plot=True,
                                       sigma=sigma)
    print('done generating from initial image with sigma = {}'.format(sigma))


def try_completing(use_cuda, sigma):
    # TODO
    return()


if __name__ == '__main__': 
    encoder_name = 'draw_models/encoder_broccoli_car_cat_7000.pth'
    decoder_name = 'draw_models/decoder_broccoli_car_cat_7000.pth'
    try_generating(0.01,
                   encoder_name,
                   decoder_name,
                   uncondition=False)
    try_generating(0.1,
                   encoder_name,
                   decoder_name,
                   uncondition=True)

