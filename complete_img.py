import argparse
import pickle
import re

import numpy as np
import torch

from utils import HParams
from utils import Model
from utils import DataLoader
from utils.visutils import make_image
from utils.batchutils import make_image_point
'''
PROBLEM NOW: which is the z latent we are to use? A random one also?
'''

use_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parser_draw = argparse.ArgumentParser(description='Drawing arguments')
    parser_draw.add_argument('--model',
                             default='broccoli_car_cat_20000.pth',
                             help='the ending of model path in draw_models')
    parser_draw.add_argument('--sigma',
                             default=1,
                             type=float,
                             help='variance when generating a point')
    parser_draw.add_argument('--nbr_point_next',
                             default=30,
                             type=int,
                             help='nbr of point continuing the draw')
    parser_draw.add_argument('--plot', action='store_true', help='plot result')
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


def from_larray_to_3array(l_array, continue_last_stroke=True):
    '''
    l_array : list of (nbr,2)
    '''
    x = []
    y = []
    z = []
    n_strokes = len(l_array)
    for id_stroke, stroke in enumerate(l_array):
        x.append(stroke[:, 0])
        y.append(stroke[:, 1])
        tab = np.zeros(len(stroke[:,0]))
        tab[-1] = 1
        if id_stroke == n_strokes-1 and continue_last_stroke:
            tab[-1] = 0
        z.append(tab)
    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)

    return np.stack([x,y,z], axis=1)


def from_3array_to_larray(array_3):
    l_idx_jump = list(np.where(array_3[:, 2] == 1)[0])

    strokes = []
    idx_old = 0
    for idx in l_idx_jump:
        stroke = array_3[idx_old:idx+1, :2].copy()
        idx_old = idx+1
        strokes.append(stroke)

    return strokes



def compute_variance(array_offsets):
    '''
    Method : Given a sequence of offset, compute the variance distinguishing
    between the jumps.
    Input : array_offsets is a (nbr,3) numpy representing (dx, dy, p)
    '''
    l_idx_jump = list(np.where(array_offsets[:, 2] == 1)[0])
    if l_idx_jump == []:
        norms = np.linalg.norm(array_offsets[:, :2], axis=1)
        return(norms.mean(), norms.std())
    else:
        mean = 0
        std = 0
        idx_old = 0
        for idx in l_idx_jump:
            norms = np.linalg.norm(array_offsets[idx_old: idx], axis=1)
            mean += norms.mean()
            std += norms.std()
            idx_old = idx

    mean = mean/len(l_idx_jump)
    std = std/len(l_idx_jump)

    return (mean, std)


def scale_stroke(array_offsets, scale):
    '''
    It put the std of each stroke to 1
    '''
    l_idx_jump = list(np.where(array_offsets[:, 2] == 1)[0])
    if l_idx_jump == []:
        return(array_offsets/scale)
    else:
        (nbr, _) = array_offsets.shape
        array_offsets_nor = np.zeros((nbr, 3))
        idx_old = 0
        for idx in l_idx_jump:
            array_offsets_nor[idx_old:idx, 0:2] = array_offsets[idx_old:idx, 0:2]/scale
            idx_old = idx
        array_offsets_nor[:, 2] = array_offsets[:, 2]
    return(array_offsets_nor)


def adjust_img(array_offsets):
        '''
        array_offsets : numpy (nbr,3) of format (x,y,p)
        '''
        img_full = array_offsets
        # from (x,y,p) to (dx,dy,p)
        img_full[1:, 0:2] = img_full[1:, 0:2] - img_full[:-1, 0:2]
        # get standard dev
        mean_full, std_full = compute_variance(img_full)
        # scale on each stroke
        img_full = scale_stroke(img_full, std_full)
        # from (dx,dy,p) to input of RNN
        img_full = make_image_point(img_full)
        return(img_full)


def complete(model_name,
             use_cuda,
             nbr_point_next,
             painting_completing=None,
             painting_conditioning=None,
             idx=None,
             sig=0.1):
    '''
    Methods:
    --------
    painting_completing/conditioning are images in format (nbr, 3) (x, y, p)
    and we want to complete them. Beware there are others format. (dx, dy, p) where
    (dx, dy) is the offset with respect to the previous point. An another one
    of size (nbr, 5), the one used by the neural net.

    Inputs:
    -------
        - model_name : a string of the type 'broccoli_car_cat_20000.pth'.
        use_cuda : Boolean.
        - idx : integer representing the index of an image to the dataset
    associated with model_name.
        - painting_completing: should be a parametred version ready for the neural network. It
    should be of the same format as 'datum'. Hence a numpy array of dimension
    (nbr_points, 3) and each line being (x,y,p) where (x,y) represents
    the coordinate (TODO: with respect to what???) and p\in{0,1} saying weither
    or not the point are linked.
        - painting_conditioning: comes in the same format. The goal is to provide
    a latent vector z.
        - sig : the variance of the latent normal vector z if not using the
    latent vector of a global image. (Although we may want to have the latent
    vector of a given area..)

    Outputs:
    --------
    '''
    if painting_completing is None and idx is None:
        raise ValueError('there should at least one of the two that is\
                         None.')
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

    # prepare img to complete and image that condition
    if idx is not None:
        # Then completing an image of the dataset
        regular = '[_][0-9]+'
        name_mid = re.split(regular, args_draw.model[:-4])
        name_mid = name_mid[0]
        path_data = 'data/' + name_mid + '.npz'
        try:
            dataloader = DataLoader(path_data, hp)
        except:
            # TODO: find which is the error to except
            print('the path to dataset is not working')
        idx = np.random.randint(1, 30)
        datum = dataloader.data[idx]
        (nbr_points_datum, _) = datum.shape
        # TODO: make 0.6 as a parameter...
        datum = datum[:int(nbr_points_datum*0.6)]
        # TODO: remember what make_image_point is doing
        img_full = make_image_point(datum)
        img_to_complete = make_image_point(datum)  # img is in the parametrized format
    else:
        # completing our own image
        # It is in format (x,y,p) put it into that (dx,dy,p) format
        datum = painting_completing
        # offset the coordinate
        datum[1:, 0:2] = datum[1:, 0:2] - datum[:-1, 0:2]
        # compute the std of initial image
        mean_ini, std_ini = compute_variance(datum)
        # normalize the painting to complete
        datum = scale_stroke(datum, std_ini)
        # format from (dx,dy,p) to the 5
        img_to_complete = make_image_point(datum)

        # determining the image that will condition the latent vector z.
        if painting_conditioning is not None:
            # format (x,y,p) to (dx,dy,p)
            img_full = painting_conditioning
            img_full[1:, 0:2] = img_full[1:, 0:2] - img_full[:-1, 0:2]
            mean_full, std_full = compute_variance(img_full)
            img_full = scale_stroke(img_full, std_full)
            img_full = make_image_point(img_full)
        else:
            img_full = None

    # complete the stuff : img_tail is in format (dx, dy, p)
    # max_length_mean = 
    img_tail = model.finish_drawing_point(img_to_complete,
                                          use_cuda,
                                          nbr_point_next=nbr_point_next,
                                          img_full=img_full,
                                          sigma=sig)
    # process the tail so that it has the same variance as the images
    # it tries to complete.
    mean_tail, std_tail = compute_variance(img_tail)
    # print(mean_tail, std_tail)

    img_tail = scale_stroke(img_tail, std_tail)

    # TODO: check that the concatenation is ok
    img_total = np.concatenate((datum, img_tail), 0)

    # plot the image..
    (img_coo, img_offset) = make_seq(img_total[:, 0],
                                     img_total[:, 1],
                                     img_total[:, 2])
    (img_tail_coo, img_tail_offset) = make_seq(img_tail[:, 0],
                                               img_tail[:, 1],
                                               img_tail[:, 2])

    make_image(img_coo, 1, dest_folder=None, name='_output_', plot=True)#plot=args_draw.plot)
    make_image(img_tail_coo, 2, dest_folder=None, name='_output_', plot=True)
    # TODO: return a list of array...


def create_example_painting():
    # create a half-circle
    painting = 0
    nbr_point_circle = 30
    angle = np.linspace(0, np.pi, nbr_point_circle)
    x = np.cos(angle)
    y = np.sin(angle)
    # x[1:] = x[1:] - x[:-1]
    # y[1:] = y[1:] - y[:-1]
    painting = np.zeros((nbr_point_circle, 3))
    painting[:, 0] = x
    painting[:, 1] = y
    return(painting)

# if __name__ == '__main__':
    # # non dx,dy,dz
    # paint_circle = create_example_painting()
    # paint_circle[10, 2] = 1
    # paint_circle[-1, 2] = 1
    # circle_as_stroke = from_3array_to_larray(paint_circle)
    # paint_circle_bis = from_larray_to_3array(circle_as_stroke)
    # import pdb; pdb.set_trace()


def tina_et_charlie(model_name,
             use_cuda,
             nbr_point_next,
             painting_completing,
             painting_conditioning,
             sig=0.1):
    # transform seq of stroke into (nbr,3)
    painting_completing = from_larray_to_3array(painting_completing)
    painting_conditioning = from_larray_to_3array(painting_conditioning)

    # load hp
    hp_path = 'draw_models/hp_folder/' + model_name[:-4] + '.pickle'
    with open(hp_path, 'rb') as handle:
        hp = pickle.load(handle)
    hp.use_cuda = use_cuda

    # load model
    model = Model(hyper_parameters=hp, parametrization='point')
    encoder_name = 'draw_models/encoder_' + model_name
    decoder_name = 'draw_models/decoder_' + model_name
    model.load(encoder_name, decoder_name)

    # It is in format (x,y,p) put it into that (dx,dy,p) format
    datum = painting_completing
    # offset the coordinate
    datum[1:, 0:2] = datum[1:, 0:2] - datum[:-1, 0:2]
    # compute the std of initial image
    mean_ini, std_ini = compute_variance(datum)
    # normalize the painting to complete
    datum = scale_stroke(datum, std_ini)
    # format from (dx,dy,p) to the 5
    img_to_complete = make_image_point(datum)

    # determining the image that will condition the latent vector z.
        # format (x,y,p) to (dx,dy,p)
    img_full = painting_conditioning
    img_full[1:, 0:2] = img_full[1:, 0:2] - img_full[:-1, 0:2]
    mean_full, std_full = compute_variance(img_full)
    img_full = scale_stroke(img_full, std_full)
    img_full = make_image_point(img_full)

    # complete
    img_tail = model.finish_drawing_point(img_to_complete,
                                          use_cuda,
                                          nbr_point_next=nbr_point_next,
                                          img_full=img_full,
                                          sigma=sig)

    # process the tail so that it has the same variance as the images
    # it tries to complete.
    mean_tail, std_tail = compute_variance(img_tail)
    img_tail = scale_stroke(img_tail, std_tail)
    img_total = np.concatenate((datum, img_tail), 0)

    # plot the image..
    (img_coo, img_offset) = make_seq(img_total[:, 0],
                                     img_total[:, 1],
                                     img_total[:, 2])
    (img_tail_coo, img_tail_offset) = make_seq(img_tail[:, 0],
                                               img_tail[:, 1],
                                               img_tail[:, 2])

    make_image(img_coo, 1, dest_folder=None, name='_output_', plot=True)#plot=args_draw.plot)
    make_image(img_tail_coo, 2, dest_folder=None, name='_output_', plot=True)

    # Transform (nbr,3) to list of array (nbr,2)
    # TODO: verifier la question de l'origine
    img_completed = from_3array_to_larray(img_tail_coo)
    return img_completed 

if __name__ == '__main__':
    paint_circle = create_example_painting()
    paint_circle[10, 2] = 0
    paint_circle[-1, 2] = 1
    paint = from_3array_to_larray(paint_circle)

    tina_et_charlie(args_draw.model,
                    use_cuda,
                    args_draw.nbr_point_next,
                    paint,
                    paint,
                    args_draw.sigma)

    # complete(args_draw.model,
             # use_cuda,
             # int(args_draw.nbr_point_next),
             # painting_completing=create_example_painting(),
             # #painting_conditioning=create_example_painting(),
             # idx=None)
