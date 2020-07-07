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

"""ResNet specific operations.

Defines the default ResNet generator and discriminator blocks and some helper
operations such as unpooling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from compare_gan.architectures import abstract_arch
from compare_gan.architectures import arch_ops as ops

from six.moves import range
import numpy as np
import tensorflow as tf


def unpool(value, name="unpool"):
  """Unpooling operation.

  N-dimensional version of the unpooling operation from
  https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
  Taken from: https://github.com/tensorflow/tensorflow/issues/2169

  Args:
    value: a Tensor of shape [b, d0, d1, ..., dn, ch]
    name: name of the op
  Returns:
    A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
  """
  with tf.name_scope(name) as scope:
    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
      out = tf.concat([out, tf.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size, name=scope)
  return out


def validate_image_inputs(inputs, validate_power2=True):
  inputs.get_shape().assert_has_rank(4)
  inputs.get_shape()[1:3].assert_is_fully_defined()
  if inputs.get_shape()[1] != inputs.get_shape()[2]:
    raise ValueError("Input tensor does not have equal width and height: ",
                     inputs.get_shape()[1:3])
  width = inputs.get_shape().as_list()[1]
  if validate_power2 and math.log(width, 2) != int(math.log(width, 2)):
    raise ValueError("Input tensor `width` is not a power of 2: ", width)


class ResNetBlock(object):
  """ResNet block with options for various normalizations."""

  def __init__(self,
               name,
               in_channels,
               out_channels,
               scale,
               is_gen_block, # this is incorrectly named. If true, it only does "scale first then none, or vice versa"
               layer_norm=False,
               spectral_norm=False,
               batch_norm=None):
    """Constructs a new ResNet block.

    Args:
      name: Scope name for the resent block.
      in_channels: Integer, the input channel size.
      out_channels: Integer, the output channel size.
      scale: Whether or not to scale up or down, choose from "up", "down" or
        "none".
      is_gen_block: Boolean, deciding whether this is a generator or
        discriminator block.
      layer_norm: Apply layer norm before both convolutions.
      spectral_norm: Use spectral normalization for all weights.
      batch_norm: Function for batch normalization.
    """
    assert scale in ["up", "down", "none"]
    self._name = name
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._scale = scale
    # In SN paper, if they upscale in generator they do this in the first conv.
    # For discriminator downsampling happens after second conv.
    self._scale1 = scale if is_gen_block else "none"
    self._scale2 = "none" if is_gen_block else scale
    self._layer_norm = layer_norm
    self._spectral_norm = spectral_norm
    self.batch_norm = batch_norm

  def __call__(self, inputs, z, y, is_training):
    return self.apply(inputs=inputs, z=z, y=y, is_training=is_training)

  def _get_conv(self, inputs, in_channels, out_channels, scale, suffix,
                kernel_size=(3, 3), strides=(1, 1)):
    """Performs a convolution in the ResNet block."""
    if inputs.get_shape().as_list()[-1] != in_channels:
      raise ValueError("Unexpected number of input channels.")
    if scale not in ["up", "down", "none"]:
      raise ValueError(
          "Scale: got {}, expected 'up', 'down', or 'none'.".format(scale))

    outputs = inputs
    if scale == "up":
      outputs = unpool(outputs)
    outputs = ops.conv2d(
        outputs,
        output_dim=out_channels,
        k_h=kernel_size[0], k_w=kernel_size[1],
        d_h=strides[0], d_w=strides[1],
        use_sn=self._spectral_norm,
        name="{}_{}".format("same" if scale == "none" else scale, suffix))
    if scale == "down":
      outputs = tf.nn.pool(outputs, [2, 2], "AVG", "SAME", strides=[2, 2],
                           name="pool_%s" % suffix)
    return outputs

  def apply(self, inputs, z, y, is_training):
    """"ResNet block containing possible down/up sampling, shared for G / D.

    Args:
      inputs: a 3d input tensor of feature map.
      z: the latent vector for potential self-modulation. Can be None if use_sbn
        is set to False.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, whether or notthis is called during the training.

    Returns:
      output: a 3d output tensor of feature map.
    """
    if inputs.get_shape().as_list()[-1] != self._in_channels:
      raise ValueError("Unexpected number of input channels.")

    with tf.variable_scope(self._name, values=[inputs]):
      output = inputs

      shortcut = self._get_conv(
          output, self._in_channels, self._out_channels, self._scale,
          suffix="conv_shortcut")

      output = self.batch_norm(
          output, z=z, y=y, is_training=is_training, name="bn1")
      if self._layer_norm:
        output = ops.layer_norm(output, is_training=is_training, scope="ln1")

      output = tf.nn.relu(output)
      output = self._get_conv(
          output, self._in_channels, self._out_channels, self._scale1,
          suffix="conv1")

      output = self.batch_norm(
          output, z=z, y=y, is_training=is_training, name="bn2")
      if self._layer_norm:
        output = ops.layer_norm(output, is_training=is_training, scope="ln2")

      output = tf.nn.relu(output)
      output = self._get_conv(
          output, self._out_channels, self._out_channels, self._scale2,
          suffix="conv2")

      # Combine skip-connection with the convolved part.
      output += shortcut
      return output
  #----------------------------------------------------------------------------
  # Modulated convolution layer.

  def _modulated_conv2d_layer(self, x, z, channels_in, channels_out, kernel, scale_up_down_none, demodulate, suffix, gain=1, use_wscale=True, lrmul=1, weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias'):
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    w = self._get_weight([kernel, kernel, channels_in, channels_out], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=suffix+'_'+weight_var)
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    z_dim = z.shape[-1].value
    style_channels_out = channels_in
    fan_in = z_dim * style_channels_out # this would need to be different if the layer sizes below were different
    he_std = gain / np.sqrt(fan_in) # He init
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul # Naming conventions from StyleGAN
    s = ops.lrelu(ops.linear(z, style_channels_out, lrmul=runtime_coef, scope=suffix+'_'+mod_weight_var, stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
    # s = dense_layer(z, fmaps=channels_in, weight_var=mod_weight_var) # [BI] Transform incoming W (latent) to style.
    # s = apply_bias_act(s, bias_var=mod_bias_var) + 1 # [BI] Add bias (initially 1).
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
      d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
      ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    #if fused_modconv:
    #  x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
    #  w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    #else:
    x *= tf.cast(s[:, np.newaxis, np.newaxis, :], x.dtype) # [BhwI] Not fused => scale input activations.

    # Convolution with optional up/downsampling.
    x = self._get_conv(x, channels_in, channels_out, scale_up_down_none, suffix, kernel_size=(kernel, kernel), strides=(1, 1))
    #if up:
    #  x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=resample_kernel)
    #elif down:
    #  x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=resample_kernel)
    #else:
    #  x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NHWC', strides=[1,1,1,1], padding='SAME')
    
    # Reshape/scale output.
    #if fused_modconv:
    #  x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    #elif demodulate:
    if demodulate:
      x *= tf.cast(d[:, np.newaxis, np.newaxis, :], x.dtype) # [BhwO] Not fused => scale output activations.
    return x

  # Single convolution layer with all the bells and whistles.
  def _style_mod_conv_layer(self, x, z, channels_in, channels_out, scale_up_down_none, suffix, bias_var='bias'):
    print('entering stylemod x shape',x.shape,'in channels',channels_in,'out channels',channels_out)
    assert x.shape[3].value == channels_in
    kernel = 3 # from StyleGAN
    demodulate = True # from StyleGAN
    # resample_kernel = [1,3,3,1] # from StyleGAN
    x = self._modulated_conv2d_layer(x, z, channels_in, channels_out, kernel, scale_up_down_none, demodulate, suffix)
    print('post-modulation shape',x.shape)
    #if randomize_noise:
    #    noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
    #else:
    #    noise = tf.cast(noise_inputs[layer_idx], x.dtype)
    #noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros(), use_resource=True)
    #x += noise * tf.cast(noise_strength, x.dtype)
    lrmul = 1
    c = x.shape[3]
    assert c == channels_out
    b = tf.get_variable(suffix+'_'+bias_var, shape=[c], initializer=tf.initializers.zeros(), use_resource=True) * lrmul
    x = x + b
    x = ops.lrelu(x)
    print('leaving stylemod x shape',x.shape)
    return x

  def _get_weight(self, shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
      fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
      he_std = gain / np.sqrt(fan_in) # He init

      # Equalized learning rate and custom learning rate multiplier.
      if use_wscale:
          init_std = 1.0 / lrmul
          runtime_coef = he_std * lrmul
      else:
          init_std = he_std / lrmul
          runtime_coef = lrmul

      # Create variable.
      init = tf.initializers.random_normal(0, init_std)
      return tf.get_variable(weight_var, shape=shape, initializer=init, use_resource=True) * runtime_coef


class ResNetGenerator(abstract_arch.AbstractGenerator):
  """Abstract base class for generators based on the ResNet architecture."""

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return ResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=True,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)


class ResNetDiscriminator(abstract_arch.AbstractDiscriminator):
  """Abstract base class for discriminators based on the ResNet architecture."""

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return ResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=False,
        layer_norm=self._layer_norm,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)
        