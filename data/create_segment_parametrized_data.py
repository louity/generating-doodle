import argparse 
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np

# from sketchrnn.paramutils import seg_to_point, point_to_seg
from data_utils import seg_to_point, point_to_seg

parser = argparse.ArgumentParser(description='translate .npz file from point param to seg param')
parser.add_argument('--file', default='cat.npz',
                    help='.npz file to reparametrize')
args = parser.parse_args()

data_location = args.file 


'''
A datum is a list of (dx, dy, p) where (dx, dy) is the offset with respect
to the previous point.

Here the goal is to transform existing cat point parametrized drawing into
cat segment drawings. This will allow (with the ordering supervision of the
cat) to simply make a sanity check on the proposed parametrization.
'''


def make_absolute(datum) -> np.ndarray:
    '''transform a datum with relative info in a datum with absolute
    coordinate'''
    x_sample = np.cumsum(datum[:, 0], 0)
    y_sample = np.cumsum(datum[:, 1], 0)
    z_sample = np.array(datum[:, 2])
    sequence = np.stack([x_sample, y_sample, z_sample]).T
    return(sequence)


debug = False
if debug:
    dataset = np.load(data_location, encoding='latin1')
    data = dataset['train']
    x1 = np.random.normal(0, 1, 2)
    x2 = np.random.normal(0, 1, 2)
    print(x1, x2)
    seg = point_to_seg(x1, x2)
    (x1_l, x2_l) = seg_to_point(seg)
    # should be (x1, x2) up to a permutation
    print(x1_l, x2_l)


def transform_datum_point_to_seg(datum, debug=False) -> np.ndarray:
    '''
    Method:
    -------
    Transform a drawing with point parametrisation (x,y,p) into line
    parametrisation, e.g. with (x,y,r,phi) segment parametrisation.
    (x,y) is the center point of the segment. r is the radius of the segment
    and phi the angle of the segment with horizon.
    Input:
    ------
    datum : numpy array of shape (nbr_point,3)
    '''
    if type(datum) != np.ndarray:
        raise ValueError('the type of datum should be np.ndarray')
    if np.shape(datum) != (np.shape(datum)[0], 3):
        raise ValueError('the shape of datum should be (nbr_points, 3)')
    # Take a non-relative version of the datum:
    sequence = make_absolute(datum)
    # Split in strokes
    point_strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0]+1)
    # remove stroke with only one point
    point_strokes = [stroke for stroke in point_strokes
                     if np.shape(stroke)[0] > 1]
    segment_strokes = []
    for stroke in point_strokes:
        segment = np.zeros((np.shape(stroke)[0]-1, 4))
        # transform the sequence of points in a sequence of line
        for i in range(np.shape(stroke)[0]-1):
            segment[i, :] = point_to_seg(stroke[i, :], stroke[i+1, :])
        segment_strokes.append(segment)

    # merge all the segments
    segments = np.concatenate(segment_strokes)
    return(segments)


if debug:
    segments = transform_datum_point_to_seg(data[0])
    print(segments[:, 3])


def make_image_seg(sequence) -> None:
    '''
    Method:
    -------
    Plot the image as a sequence of segments
    Input:
    ------
    sequence : an array of dimension (nbr_points, 4) that parametrizes the img
    as a sequence of segments
    '''
    if np.shape(sequence)[1] != 4:
        raise ValueError('we expect an image parametrized as a list of segment')
    lines = []
    for i_line, line in enumerate(sequence):
        (x1, x2) = seg_to_point(line)
        line = [x1, x2]
        lines.append(line)

    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)


def make_image_point(datum, offset=False) -> None:
    '''
    datum should be an array of dimension (nbr_point, 3). If offset, it means
    that datum[i,:2] is the offset of the i^{th} with respect to the previous
    one, otherwise it represents some king of coordinate.
    '''
    if np.shape(datum)[1] != 3:
        raise ValueError('we expect an image parametrized as a list of point')
    if not offset:
        sequence = make_absolute(datum)
    else:
        sequence = datum
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0]+1)
    plt.figure(1)
    for s in strokes:
        plt.plot(s[:, 0], s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()


def transform_point_to_seg_npz(file_loc) -> None:
    '''
    Transforms a npz file containing data in a point representation into the
    same npz with segment representation.
    Trying to preserve the initial structure of the file.
    '''
    # dissociate file_name from file_folder..
    if file_loc[-4:] != '.npz':
        raise ValueError('your file name is not finishing with .npz')
    file_name = file_loc[:-4]
    outfile = file_name + '_segment.npz'
    # load point parametrized .npz file
    # dataset = np.load(data_location, encoding='latin1')
    dataset = np.load(file_loc, encoding='latin1')
    # save as a .npz
    l_name = ['train', 'test', 'valid']
    if set(l_name) != set(['train', 'test', 'valid']):
        raise ValueError('the input .npz file should contain train,\
                          test and valid')
    train = np.array([transform_datum_point_to_seg(datum) for
                     datum in dataset['train']])
    test = np.array([transform_datum_point_to_seg(datum) for
                     datum in dataset['test']])
    valid = np.array([transform_datum_point_to_seg(datum) for
                     datum in dataset['valid']])
    np.savez(outfile, train=train, test=test, valid=valid)


if debug:
    segments = transform_datum_point_to_seg(data[2])
    make_image_seg(segments)
    make_image_point(data[2])
    plt.show()

    # Uncomment if you want to create the file cat_segment.npz
    transform_point_to_seg_npz('cat.npz')

    data_location = 'cat_segment.npz'
    dataset = np.load(data_location, encoding='latin1')
    data = dataset['train']
    sequence = data[200]
    make_image_seg(sequence)

if __name__ == '__main__':
    transform_point_to_seg_npz(data_location)

