# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tensorflow as tf
import os, math


def check_folder(log_dir):
  if not tf.gfile.IsDirectory(log_dir):
    tf.gfile.MakeDirs(log_dir)
  return log_dir

# drange_in could be something like [-1, 1] and drange_out could be [0, 255] or vice versa
def adjust_dynamic_range(data, drange_in, drange_out):
  if drange_in != drange_out:
    scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
    bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
    data = data * scale + bias
  return data

def save_images(images, image_path):
  with tf.gfile.Open(image_path, "wb") as f:
    scipy.misc.imsave(f, images * 255.0)


def rotate_images(images, rot90_scalars=(0, 1, 2, 3)):
  """Return the input image and its 90, 180, and 270 degree rotations."""
  images_rotated = [
      images,  # 0 degree
      tf.image.flip_up_down(tf.image.transpose_image(images)),  # 90 degrees
      tf.image.flip_left_right(tf.image.flip_up_down(images)),  # 180 degrees
      tf.image.transpose_image(tf.image.flip_up_down(images))  # 270 degrees
  ]

  results = tf.stack([images_rotated[i] for i in rot90_scalars])
  results = tf.reshape(results,
                       [-1] + images.get_shape().as_list()[1:])
  return results


def matrix_transform(dimensions, image, alpha=0.1, zoomopt='zoomin', transopt='xy', rotate=False, shear=False):
    """
    Does various matrix based transformations of a 3-D image tensor.
    Pads edges with the values at that edge in the original image.
    :param image: 3-D tensor with a single image.
    :param alpha: Strength of augmentation.
    :param zoomopt: Options for zoom. 'zoomin', 'zoomout', or 'none'.
    :param transopt: Options for translation. 'x', 'y', or 'xy'.
    :param rotate: if True, randomly rotate image.
    :param shear: if True, randomly shear image.
    :return: 3-D tensor with a single image.
    """
    dimensions = int(dimensions)
    XDIM = dimensions % 2  # fix for size 331

    if rotate:
        rot = 15. * tf.random.normal([1], dtype='float32') * alpha
    else:
        rot = tf.constant([0], dtype='float32')

    if shear:
        shr = 5. * tf.random.normal([1], dtype='float32') * alpha
    else:
        shr = tf.constant([0], dtype='float32')

    if zoomopt == 'zoomin':
        h_zoom = 1. + tf.random.uniform([1], dtype='float32') * alpha
        w_zoom = h_zoom
    elif zoomopt == 'zoomout':
        h_zoom = 1. - tf.random.uniform([1], dtype='float32') * alpha
        w_zoom = h_zoom
    elif zoomopt == 'both':
        h_zoom = 1. - tf.random.uniform([1], minval=1-alpha, maxval=1+alpha, dtype='float32') * alpha
        w_zoom = h_zoom
    else:
        h_zoom = tf.constant([1.])
        w_zoom = h_zoom

    if 'y' in transopt:
        # TODO: not sure these values are right
        h_shift = (tf.random.uniform([1], dtype='float32') - 0.5) * dimensions * alpha
    else:
        h_shift = tf.constant([0.])

    if 'x' in transopt:
        w_shift = (tf.random.uniform([1], dtype='float32') - 0.5) * dimensions * alpha
    else:
        w_shift = tf.constant([0.])

    # GET TRANSFORMATION MATRIX
    m = get_matrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(dimensions // 2, -dimensions // 2, -1), dimensions)
    y = tf.tile(tf.range(-dimensions // 2, dimensions // 2), [dimensions])
    z = tf.ones([dimensions * dimensions], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = tf.tensordot(m, tf.cast(idx, dtype='float32'), 1)
    idx2 = tf.cast(idx2, dtype='int32')
    idx2 = tf.clip_by_value(idx2, -dimensions // 2 + XDIM + 1, dimensions // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([dimensions // 2 - idx2[0,], dimensions // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [dimensions, dimensions, 3])

def get_matrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    returns 3x3 transform matrix which transforms indices in the image tensor
    """

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(
        tf.concat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(tf.concat([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0),
                              [3, 3])

    return tf.tensordot(tf.tensordot(rotation_matrix, shear_matrix, 1), tf.tensordot(zoom_matrix, shift_matrix, 1), 1)

def cutmix_binary_class_masks(reals, fakes, precomputed_masks=None):
  batch_size = reals.shape[0]
  h = int(reals.shape[1])
  w = int(reals.shape[2])
  c = int(reals.shape[3])
  assert c == 3 or c == 1 or c == 4
  
  out_images = []
  out_masks = []

  #indexed_reals = [reals[i] for i in range(batch_size)]
  #indexed_fakes = [fakes[i] for i in range(batch_size)]
  
  for i in range(batch_size):

    real = reals[i, :] # indexed_reals[i] # reals[i, :]
    fake = fakes[i, :] # indexed_fakes[i] # fakes[i, :]

    out_mask = None
    if precomputed_masks is not None:
      out_mask = precomputed_masks[i, :]
    else:
      alpha = 0.98
      y = math.ceil((h * alpha) / 2.) * 2
      x = math.ceil((w * alpha) / 2.) * 2
      center = tf.zeros([x, y, c]) #tf.cast(tf.fill([x, y, channel_count], one_or_zero), tf.float32)
      pad_y = tf.cast((h - y) / 2, dtype=tf.int32)
      pad_x = tf.cast((w - x) / 2, dtype=tf.int32)
      paddings = [[pad_y, pad_y], [pad_x, pad_x], [0, 0]]
      padded_center = tf.pad(center, paddings, constant_values=1)
      dimensions = h # hack
      pad_mask = 1 - matrix_transform(dimensions, padded_center, alpha=1, zoomopt='zoomout', transopt='xy', rotate=True, shear=True)
      pad_mask = tf.reshape(tf.reduce_mean(pad_mask, axis=2), (1, h, w, 1))
      summation = tf.reduce_sum(pad_mask)
      area = h * w
      percent_covered = summation / area
      inv_percent_covered = (area - summation) / area
      one_or_zero = tf.round(percent_covered)
      zero_or_one = tf.round(inv_percent_covered)
      pad_mask = ((1 - pad_mask) * one_or_zero) + ((pad_mask) * zero_or_one)
      out_mask = pad_mask

    # create the image masked out
    out_image = real * out_mask + fake * (1 - out_mask)
    out_images.append(out_image)
    out_masks.append(out_mask)
          
  out_images = tf.reshape(tf.stack(out_images),(batch_size, h, w, c))
  out_masks = tf.reshape(tf.stack(out_masks),(batch_size, h, w, 1))
  return out_images, out_masks



def gaussian(batch_size, n_dim, mean=0., var=1.):
  return np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
