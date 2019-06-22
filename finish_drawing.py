import argparse
import pickle
import re

import numpy as np
import torch

from sketch_rnn import HParams
from sketch_rnn import Model
from sketch_rnn import DataLoader
from sketchrnn.visutils import make_image
from sketchrnn.batchutils import make_image_point
'''
Run python finish_drawing.py --experiment uncondition --sigma 0.01
to see ouput of unconditionned generation with a given value of sigma.
uncondition means that we simply pick a random gaussian vector and feed it to the 
decoder without any latent-vector z representing an image. 
'''

use_cuda = torch.cuda.is_available()
# hp.use_cuda = use_cuda

parser_draw = argparse.ArgumentParser(description='Drawing arguments')
parser_draw.add_argument('--model',
                         default='broccoli_car_cat_20000.pth',
                         help='the ending of the model path in draw_models')
parser_draw.add_argument('--sigma',
                         default=1,
                         help='variance when generating a point')
parser_draw.add_argument('--experiment',
                         default='uncondition',
                         choices=['uncondition', 'complete'],
                         help='mode of drawing')
parser_draw.add_argument('--nbr_point_next',
                         default=30,
                         help='nbr of point continuing the draw')
args_draw = parser_draw.parse_args()


def make_seq(seq_x, seq_y, seq_z):
    '''
    To go from offset to plain coordinate
    '''
    x_sample = np.cumsum(seq_x, 0)
    y_sample = np.cumsum(seq_y, 0)
    z_sample = np.array(seq_z)
    sequence_coo = np.stack([x_sample, y_sample, z_sample]).T
    sequence_offset = np.stack([np.array(seq_x),
                               np.array(seq_y), np.array(z_sample)]).T
    return(sequence_coo, sequence_offset)


# load hp
hp_path = 'draw_models/hp_folder/' + args_draw.model[:-4] + '.pickle'
with open(hp_path, 'rb') as handle:
    hp = pickle.load(handle)
hp.use_cuda = use_cuda
# load model
model = Model(hyper_parameters=hp, parametrization='point')
encoder_name = 'draw_models/encoder_' + args_draw.model
decoder_name = 'draw_models/decoder_' + args_draw.model
model.load(encoder_name, decoder_name)


if args_draw.experiment not in ['uncondition', 'complete']:
    raise ValueError('experiment should either be uncondition or complete')

if args_draw.experiment == 'uncondition':
    model.conditional_generation_point(10, uncondition=True, plot=True,
                                       sigma=float(args_draw.sigma))
    # 10 is arbitrary.
elif args_draw.experiment == 'complete':
    # TODO: process the image to complete
    regular = '[_][0-9]+'
    name_mid = re.split(regular, args_draw.model[:-4])
    name_mid = name_mid[0]
    path_data = 'data/' + name_mid + '.npz'
    dataloader = DataLoader(path_data, hp)
    idx = np.random.randint(0, 30)
    datum = dataloader.data[idx]
    (nbr_points_datum, _) = datum.shape
    # TODO: make 0.6 as a parameter...
    datum = datum[:int(nbr_points_datum*0.6)]
    # TODO: remember what make_image_point is doing
    img_full = make_image_point(datum)
    img_to_complete = make_image_point(datum)  # img is in the parametrized format
    # finish the drawing
    img_tail = model.finish_drawing_point(img_to_complete,
                                          use_cuda,
                                          nbr_point_next=int(args_draw.nbr_point_next),
                                          img_full=img_full)
    img_total = np.concatenate((datum, img_tail), 0)
    # TODO: plot with a distinct color what is added
    (img_coo, img_offset) = make_seq(img_total[:, 0],
                                     img_total[:, 1],
                                     img_total[:, 2])
    (img_tail_coo, img_tail_offset) = make_seq(img_tail[:, 0],
                                               img_tail[:, 1],
                                               img_tail[:, 2])

    make_image(img_coo, 1, dest_folder=None, name='_output_', plot=True)
    make_image(img_tail_coo, 2, dest_folder=None, name='_output_', plot=True)
