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

"""Implementation of popular GAN losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan import utils
import gin
import tensorflow as tf


def check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits):
  """Checks the shapes and ranks of logits and prediction tensors.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].

  Raises:
    ValueError: if the ranks or shapes are mismatched.
  """
  def _check_pair(a, b):
    if a != b:
      raise ValueError("Shape mismatch: %s vs %s." % (a, b))
    if len(a) != 2 or len(b) != 2:
      raise ValueError("Rank: expected 2, got %s and %s" % (len(a), len(b)))

  if (d_real is not None) and (d_fake is not None):
    _check_pair(d_real.shape.as_list(), d_fake.shape.as_list())
  if (d_real_logits is not None) and (d_fake_logits is not None):
    _check_pair(d_real_logits.shape.as_list(), d_fake_logits.shape.as_list())
  if (d_real is not None) and (d_real_logits is not None):
    _check_pair(d_real.shape.as_list(), d_real_logits.shape.as_list())


@gin.configurable(whitelist=[])
def non_saturating(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Non-saturating loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("non_saturating_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_logits, labels=tf.ones_like(d_real_logits),
        name="cross_entropy_d_real"))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits),
        name="cross_entropy_d_fake"))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.ones_like(d_fake_logits),
        name="cross_entropy_g"))
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def pixel_cross_entropy(d_real_logits, d_fake_logits, d_real=None, d_fake=None, cutmix_masks=None):
  """Returns the discriminator and generator loss for sigmoid cross-entropy.

  Args:
    d_real_logits: logits for real 1 channel HxW image (mask) output, shape [batch_size, H, W, 1].
    d_fake_logits: logits for fake 1 channel HxW image (mask) output, shape [batch_size, H, W, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("pixel_cross_entropy"):
    # incoming mask logit shape should be [batch_size, H, W, 1] 
    assert len(d_real_logits.shape) == 4 and d_real_logits.shape[3] == 1
    assert len(d_fake_logits.shape) == 4 and d_fake_logits.shape[3] == 1
    b, h, w = d_real_logits.shape[0], d_real_logits.shape[1], d_real_logits.shape[2]
    # flat_shape will be [batch_size * H * W, 1]
    flat_shape = [b * h * w, 1]
    d_real_logits_flat = tf.reshape(d_real_logits, flat_shape)
    d_fake_logits_flat = tf.reshape(d_fake_logits, flat_shape)
    d_real_log_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_logits_flat,
        labels=tf.ones_like(d_real_logits_flat),
        name="cross_entropy_d_real"
    )
    d_fake_log_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits_flat,
        labels=tf.zeros_like(d_fake_logits_flat),
        name="cross_entropy_d_fake"
    )
    g_log_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits_flat,
        labels=tf.ones_like(d_fake_logits_flat),
        name="cross_entropy_g"
    )
    # reshaped log loss to be [batch_size, H, W, 1] 
    d_real_log_loss = tf.reshape(d_real_log_loss, d_real_logits.shape)
    d_fake_log_loss = tf.reshape(d_fake_log_loss, d_fake_logits.shape)
    g_log_loss      = tf.reshape(g_log_loss,        d_fake_logits.shape)
    sum_d_real_log_loss = tf.reduce_sum(tf.reduce_sum(d_real_log_loss, axis=2), axis=1)
    sum_d_fake_log_loss = tf.reduce_sum(tf.reduce_sum(d_fake_log_loss, axis=2), axis=1)
    sum_g_log_loss      = tf.reduce_sum(tf.reduce_sum(g_log_loss,      axis=2), axis=1)
    # sum loss reduced shape should be [batch_size, 1] 
    g_loss      = tf.cast(tf.reduce_mean(sum_g_log_loss),      tf.float32) / tf.cast(h * w, tf.float32)
    d_loss_real = tf.cast(tf.reduce_mean(sum_d_real_log_loss), tf.float32) / tf.cast(h * w, tf.float32)
    d_loss_fake = tf.cast(tf.reduce_mean(sum_d_fake_log_loss), tf.float32) / tf.cast(h * w, tf.float32)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def wasserstein(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Wasserstein loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("wasserstein_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = -tf.reduce_mean(d_real_logits)
    d_loss_fake = tf.reduce_mean(d_fake_logits)
    d_loss = d_loss_real + d_loss_fake
    g_loss = -d_loss_fake
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def least_squares(d_real, d_fake, d_real_logits=None, d_fake_logits=None):
  """Returns the discriminator and generator loss for the least-squares loss.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: ignored.
    d_fake_logits: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("least_square_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.square(d_real - 1.0))
    d_loss_fake = tf.reduce_mean(tf.square(d_fake))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    g_loss = 0.5 * tf.reduce_mean(tf.square(d_fake - 1.0))
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def hinge(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for the hinge loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("hinge_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
    d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
    d_loss = d_loss_real + d_loss_fake
    g_loss = - tf.reduce_mean(d_fake_logits)
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def pixel_hinge(d_real_logits, d_fake_logits, d_real=None, d_fake=None, cutmix_masks=None):
  """Returns the discriminator and generator loss for pixel-wise hinge loss.

  Args:
    d_real_logits: logits for real 1 channel HxW image (mask) output, shape [batch_size, H, W, 1].
    d_fake_logits: logits for fake 1 channel HxW image (mask) output, shape [batch_size, H, W, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("pixel_hinge"):

    # incoming mask logit shape should be [batch_size, H, W, 1] 
    assert len(d_real_logits.shape) == 4 and d_real_logits.shape[3] == 1
    assert len(d_fake_logits.shape) == 4 and d_fake_logits.shape[3] == 1
    b, h, w = d_real_logits.shape[0], d_real_logits.shape[1], d_real_logits.shape[2]
    # flat_shape will be [batch_size * H * W, 1]
    flat_shape = [b * h * w, 1]
    d_real_logits_flat = tf.reshape(d_real_logits, flat_shape)
    d_fake_logits_flat = tf.reshape(d_fake_logits, flat_shape)

    # cutmix mask is 0 for fake, 1 for real, only used in generated images
    if cutmix_masks is not None:
      cutmix_masks_flat = tf.reshape(cutmix_masks, flat_shape)
      # only for the fake logits, we add a penalty (-1.0) when the cutmix mask is real (==1.0)
      d_fake_logits_flat += cutmix_masks_flat * -1.0
    d_loss_real_flat = tf.nn.relu(1.0 - d_real_logits_flat)
    d_loss_fake_flat = tf.nn.relu(1.0 + d_fake_logits_flat)
    
    # reshaped raw loss to be [batch_size, H, W, 1] 
    d_real_raw_loss = tf.reshape(d_loss_real_flat, d_real_logits.shape)
    d_fake_raw_loss = tf.reshape(d_loss_fake_flat, d_fake_logits.shape)
    sum_d_real_raw_loss = tf.reduce_sum(tf.reduce_sum(d_real_raw_loss, axis=2), axis=1)
    sum_d_fake_raw_loss = tf.reduce_sum(tf.reduce_sum(d_fake_raw_loss, axis=2), axis=1)
    sum_g_logits        = tf.reduce_sum(tf.reduce_sum(d_fake_logits,   axis=2), axis=1)
    # sum loss reduced shape should be [batch_size, 1] 

    g_loss      =  tf.cast(-tf.reduce_mean(sum_g_logits),        tf.float32) / tf.cast(h * w, tf.float32)
    d_loss_real =  tf.cast( tf.reduce_mean(sum_d_real_raw_loss), tf.float32) / tf.cast(h * w, tf.float32)
    d_loss_fake =  tf.cast( tf.reduce_mean(sum_d_fake_raw_loss), tf.float32) / tf.cast(h * w, tf.float32)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def pixel_consistency_l2norm(
    source,
    target,
  ):
  """Returns the discriminator consistency loss for pixel-wise L2 norm difference in cutmix.
  """
  with tf.name_scope("pixel_consistency_l2norm"):

    # incoming mask shape should be [batch_size, H, W, 1] 
    assert len(source.shape) == 4 and source.shape[3] == 1
    assert len(target.shape) == 4 and target.shape[3] == 1
    
    d_consistency_loss = 0.0
    diff = target - source
    sqrd = tf.square(diff)
    sumd = tf.reduce_sum(sqrd, axis=[-3,-2,-1])
    root = tf.sqrt(sumd + 1e-12)
    mean = tf.reduce_mean(root)
    d_consistency_loss = mean
    
    return d_consistency_loss


@gin.configurable("loss", whitelist=["fn"])
def get_losses(fn=non_saturating, **kwargs):
  """Returns the losses for the discriminator and generator."""
  return utils.call_with_accepted_args(fn, **kwargs)
