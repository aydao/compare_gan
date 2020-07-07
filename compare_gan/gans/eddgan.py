#
# Based on ssgan, see ssgan.py file for that code's copyright notice
#

"""Encoder-Decoder-Discriminator GAN (e.g., U-net Disc. GAN) with fake/real pixel-level loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from compare_gan.architectures.arch_ops import linear
from compare_gan.gans import loss_lib
from compare_gan.gans import modular_gan
from compare_gan.gans import penalty_lib
from compare_gan.gans import utils

import gin
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

FLAGS = flags.FLAGS

# pylint: disable=not-callable
@gin.configurable(blacklist=["kwargs"])
class EDDGAN(modular_gan.ModularGAN):
  """Encoder-Decoder-Discriminator GAN.
  """

  def __init__(self,
               pixel_loss_fn,
               cutmix,
               consistency_loss,
               consistency_loss_lambda,
               pixel_consistency_loss_fn,
               **kwargs):
    super(EDDGAN, self).__init__(**kwargs)
    self._epsilon = 1e-07
    self._pixel_loss_fn = pixel_loss_fn
    self._cutmix = cutmix
    self._consistency_loss = consistency_loss
    self._consistency_loss_lambda = consistency_loss_lambda # try 0.01
    self._pixel_consistency_loss_fn = pixel_consistency_loss_fn
    self._drange_masks = [0, 1] # from sigmoid, at least in standard U-net disc
    # To safe memory ModularGAN supports feeding real and fake samples
    # separately through the discriminator. EDDGAN does not support this to
    # avoid additional additional complexity in create_loss().
    assert not self._deprecated_split_disc_calls, \
        "Splitting discriminator calls is not supported in EDDGAN."

  def _convert_to_logits(self, y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/keras/backend.py#L4455
      y_pred = tf.clip_by_value(y_pred, self._epsilon, 1 - self._epsilon)
      return tf.log(y_pred / (1 - y_pred))

  def _i(self, x): return tf.transpose(x, [0,2,3,1]) # NCHW to NHWC
  def _o(self, x): return tf.transpose(x, [0,3,1,2]) # NHWC to NCHW

  def _add_masks_to_summary(self, masks, summary_name, params):
    # All summary tensors are synced to host 0 on every step. To avoid sending
    # more images then needed we transfer at most `sampler_per_replica` to
    # create a 8x8 image grid.
    
    # masks = self._o(masks)
    batch_size_per_replica = masks.shape[0].value
    num_replicas = params["context"].num_replicas if "context" in params else 1
    grid_shape = (self.options.get("image_grid_width", 3), self.options.get("image_grid_height", 3))
    total_num_images = batch_size_per_replica * num_replicas
    sample_num_images = np.prod(grid_shape)
    if total_num_images >= sample_num_images:
      samples_per_replica = int(np.ceil(sample_num_images / num_replicas))
    else:
      samples_per_replica = batch_size_per_replica
    image_shape = self._dataset.image_shape[:]
    sample_res = self.options.get("image_grid_resolution", 1024) # TODO: remove this.
    sample_shape = [sample_res, sample_res, image_shape[2]]
    def _merge_masks_to_grid(all_images):
      all_images = all_images[:np.prod(grid_shape)]
      shape = image_shape
      tf.logging.info('Shape                         %s',shape)
      if shape[0] > sample_shape[0] or shape[1] > sample_shape[1]:
        tf.logging.info('autoimages(%s, %s): Downscaling sampled images from %dx%d to %dx%d',
                        repr(summary_name), repr(all_images),
                        shape[0], shape[1],
                        sample_shape[0], sample_shape[1])
        all_images = tf.image.resize(all_images, sample_shape[0:2], method=tf.image.ResizeMethod.AREA)
        shape = sample_shape
      

      # NOTE: temp disabled due to debugging
      # TODO: verify the dynamic range of inputs here, some may be on [-1, 1] and not on [0, 1] 
      # all_images = utils.adjust_dynamic_range(all_images, self._drange_masks, self._drange_images)
      
      
      return tfgan.eval.image_grid(
        all_images,
        grid_shape=grid_shape,
        image_shape=shape[:2],
        num_channels=1)
    self._tpu_summary.image(summary_name,
                            masks[:samples_per_replica],
                            reduce_fn=_merge_masks_to_grid)

  def create_loss(self, features, labels, params, is_training=True):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss and self.g_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      labels: Tensor will labels. These are class indices. Use
          self._get_one_hot_labels(labels) to get a one hot encoded tensor.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    images = features["images"]  # Input images.
    generated = features["generated"]  # Fake images.
    if self.conditional:
      y = self._get_one_hot_labels(labels)
      sampled_y = self._get_one_hot_labels(features["sampled_labels"])
      all_y = tf.concat([y, sampled_y], axis=0)
    else:
      y = None
      sampled_y = None
      all_y = None

    assert not (self._consistency_loss and not self._cutmix)
    
    cutmix_masks = None
    mixed_generated = None
    if self._cutmix:
      mixed_generated, cutmix_masks = utils.cutmix_binary_class_masks(images, generated)
      # if we're doing cutmix but not cons. loss, use cutmix output as the fakes
      if not self._consistency_loss:
        generated = mixed_generated
        mixed_generated = None
    assert mixed_generated is None or self._consistency_loss

    # Compute discriminator output for real and fake images in one batch.
    all_images = tf.concat([images, generated], axis=0)
    
    # Special case: add mixed fakes to the batch, repeating the fakes' labels 
    if self._consistency_loss:
      all_images = tf.concat([all_images, mixed_generated], axis=0)
      if self.conditional:
        all_y = tf.concat([all_y, sampled_y], axis=0)

    d_all, d_all_logits, tuple_h_mask_logits = self.discriminator(
        all_images,
        y=all_y,
        is_training=is_training
    )

    # Unpack the extra returned values for the second disc head
    h, d_mask_all, d_mask_all_logits = tuple_h_mask_logits
    
    # Split apart reals and fakes
    d_real,      d_fake,      d_mixed      = None, None, None
    d_mask_real, d_mask_fake, d_mask_mixed = None, None, None
    if self._consistency_loss:
      d_real,      d_fake,      d_mixed      = tf.split(d_all,      3)
      d_mask_real, d_mask_fake, d_mask_mixed = tf.split(d_mask_all, 3)
    else:
      d_real,      d_fake,     = tf.split(d_all,      2)
      d_mask_real, d_mask_fake = tf.split(d_mask_all, 2)

    # Visualize the learned mask 
    self._add_masks_to_summary(d_mask_fake, "masks_fake", params)
    self._add_masks_to_summary(d_mask_real, "masks_real", params)
    if d_mask_mixed is not None:
      self._add_masks_to_summary(d_mask_mixed, "masks_mixed", params)
    # and add images for whichever variable is using cutmix
    if self._cutmix:
      fake_cutmixed = generated if mixed_generated is None else mixed_generated
      self._add_images_to_summary(fake_cutmixed, "fake_z_cutmixed", params)

    # Split apart the real and fake logits
    d_real_logits,      d_fake_logits,      d_mixed_logits      = None, None, None
    d_mask_real_logits, d_mask_fake_logits, d_mask_mixed_logits = None, None, None
    if self._consistency_loss:
      d_real_logits,      d_fake_logits,      d_mixed_logits      = tf.split(d_all_logits,      3)
      d_mask_real_logits, d_mask_fake_logits, d_mask_mixed_logits = tf.split(d_mask_all_logits, 3)
    else:
      d_real_logits,      d_fake_logits      = tf.split(d_all_logits,      2)
      d_mask_real_logits, d_mask_fake_logits = tf.split(d_mask_all_logits, 2)

    # Begin loss computation
    self.d_loss = 0
    self.g_loss = 0

    # Compute D_Enc loss and G_from_D_Enc loss
    d_enc_loss, _, _, g_loss_from_d_enc = loss_lib.get_losses(
        d_real=d_real,
        d_fake=d_fake,
        d_real_logits=d_real_logits,
        d_fake_logits=d_fake_logits
    )

    d_dec_consistency_loss = 0.0
    # special case
    if self._consistency_loss:

      # Will be computing norm for
      # L^cons = || D^U_dec(mix(real, fake, mask)) - mix(D^U_dec(real), D^U_dec(fake), mask) ||
      #        = || d_dec_on_cutmix - mix(d_dec_real, d_dec_fake, mask) ||
      #        = || d_dec_on_cutmix - mixed_d_dec_real_and_fake ||
      # so
      # we need 
      #     || Image3 - MixOfImages ||
      #     || Image3 - mix(Image1, Image2) ||
      #     where
      #     Image1 = D's decoding of a real image
      #     Image2 = D's decoding of a fake image
      #     Image3 = D's decoding of a cutmix image
      
      # should we be using logits?
      # no, the sigmoid puts output b/w 0 and 1, like cutmix_masks
      mix_of_d_real_and_d_fake, _ = utils.cutmix_binary_class_masks(
        d_mask_real,
        d_mask_fake,
        precomputed_masks=cutmix_masks)

      # compute consistency loss
      d_dec_consistency_loss = loss_lib.get_losses(
          fn=self._pixel_consistency_loss_fn,
          source=d_mask_mixed,
          target=mix_of_d_real_and_d_fake
      )
      # if doing consistency loss,
      # we will NOT use cutmix masks for the main loss computation
      cutmix_masks = None 
      # else if we weren't doing consistency loss,
      # but were still doing cutmix,
      # cutmix_masks would not be None 
      self._add_masks_to_summary(d_mask_mixed, "z_consistency/source", params)
      self._add_masks_to_summary(mix_of_d_real_and_d_fake, "z_consistency/target", params)

    # Compute D_Dec loss and G_from_D_Dec loss
    d_dec_loss, _log_only_d_mask_loss_real, _log_only_d_mask_loss_fake, g_loss_from_d_dec = loss_lib.get_losses(
        fn=self._pixel_loss_fn,
        d_real=d_mask_real,
        d_fake=d_mask_fake,
        d_real_logits=d_mask_real_logits,
        d_fake_logits=d_mask_fake_logits,
        cutmix_masks=cutmix_masks # None if cutmix is False or consistency_loss is True
    )

    # L_D^U <- L_D_Enc + L_D_Dec 
    self.d_loss = 0.5 * (d_enc_loss + d_dec_loss)

    # L_D^U <- L_D^U + lambda*L_D_Dec_Consistency (which is 0.0 if not enabled)
    self.d_loss += self._consistency_loss_lambda * d_dec_consistency_loss

    # L_G <- L_G_from_D_Enc + L_G_from_D_dec
    self.g_loss = 0.5 * (g_loss_from_d_enc + g_loss_from_d_dec)

    # Add penalty
    penalty_loss = penalty_lib.get_penalty_loss(
        x=images, x_fake=generated, y=y, is_training=is_training,
        discriminator=self.discriminator)
    self.d_loss += self._lambda * penalty_loss

    name = "loss_d/"
    self._tpu_summary.scalar(name + "enc_loss", d_enc_loss)
    self._tpu_summary.scalar(name + "dec_loss", d_dec_loss)
    self._tpu_summary.scalar(name + "dec_loss_fake", _log_only_d_mask_loss_fake)
    self._tpu_summary.scalar(name + "dec_loss_real", _log_only_d_mask_loss_real)
    self._tpu_summary.scalar(name + "dec_consistency_loss", d_dec_consistency_loss)
    self._tpu_summary.scalar(name + "penalty", penalty_loss)
    name = "loss_g/"
    self._tpu_summary.scalar(name + "from_enc_loss", g_loss_from_d_enc)
    self._tpu_summary.scalar(name + "from_dec_loss", g_loss_from_d_dec)

