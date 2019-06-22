import os

from matplotlib import pyplot as plt
from matplotlib import collections as mc
import numpy as np
import PIL

from sketchrnn.paramutils import seg_to_point


def make_image(sequence, epoch, dest_folder=None,
               name='_output_', plot=False) -> None:
    """
    plot drawing with separated strokes. Sequence represents already the
    coordinate?
    """
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0]+1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    name = '{}_{}.jpg'.format(epoch, name)
    if dest_folder is not None:
        os.makedirs(dest_folder, exist_ok=True)
        name = os.path.join(dest_folder, name)
    pil_image.save(name, "JPEG")
    if plot:
        plt.show()
    else:
        plt.close("all")


def make_image_seq(sequence, epoch, dest_folder='images/train_line') -> None:
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
    # saving it
    name = dest_folder + '\output_{}.jpg'.format(epoch)
    plt.savefig(name)


def plot_sketch(sequence, parametrization):
    """Plot sketch in the data in the input format."""
    # TODO: beware input format for point is the offset, not the points
    # here you are considering that the center parametrization of the
    if parametrization == 'line':
        lines = []
        for i_line, line in enumerate(sequence):
            points = seg_to_point(line)
            lines.append(points)

        lc = mc.LineCollection(lines, linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        plt.show()
    elif parametrization == 'point':
        # TODO: check if it works
        strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0]+1)
        plt.figure()
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        plt.show()
