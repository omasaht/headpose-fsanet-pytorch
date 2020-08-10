"""
Zoom Transformation
Original Implementation from: keras-preprocessing
https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/affine_transformations.py
------------------------------------------------
Modified by: Omar Hassan
August, 2020
"""

import numpy as np
import scipy
from scipy import ndimage


def _apply_random_zoom(x, zoom_range):
    """
    Apply zoom transformation given a set of range and input image.
    :param x: input image
    :param zoom_range: list of zoom range i.e. [0.8,1.2]
    :return: zoom augmented image
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: %s' % (zoom_range,))

    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = _apply_affine_transform(x, zx=zx, zy=zy)
    return x


def _transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def _apply_affine_transform(x,zx, zy):
    """
    Applies affine transformation with scale param of affine matrix
    set to zoom parameters.
    :param x: input image
    :param zx: horizontal zoom scale
    :param zy: vertical zoom scale
    :return: affine transformed input image
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    channel_axis = 2
    order = 1
    fill_mode = 'nearest'
    cval = 0

    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[:2]
    transform_matrix = _transform_matrix_offset_center(
        zoom_matrix, h, w)
    x = np.moveaxis(x, channel_axis, 0) #bring channel to first axis
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [ndimage.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=order,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.moveaxis(x, 0, channel_axis) #bring channel to last axis

    return x
