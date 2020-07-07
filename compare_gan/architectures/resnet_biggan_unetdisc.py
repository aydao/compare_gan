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

"""Re-implementation of BigGAN architecture.

Disclaimer: We note that this is our best-effort re-implementation and stress
that even minor implementation differences may lead to large differences in
trained models due to sensitivity of GANs to optimization hyperparameters and
details of neural architectures. That being said, this code suffices to
reproduce the reported FID on ImageNet 128x128.

Based on "Large Scale GAN Training for High Fidelity Natural Image Synthesys",
Brock A. et al., 2018 [https://arxiv.org/abs/1809.11096].

Supported resolutions: 32, 64, 128, 256, 512, 1024. The location of the self-attention
block must be set in the Gin config. See below.

Notable differences to resnet5.py:
- Much wider layers by default.
- 1x1 convs for shortcuts in D and G blocks.
- Last BN in G is unconditional.
- tanh activation in G.
- No shortcut in D block if in_channels == out_channels.
- sum pooling instead of mean pooling in D.
- Last block in D does not downsample.

Information related to parameter counts and Gin configuration:
128x128
-------
Number of parameters: (D) 87,982,370 (G) 70,433,988
Required Gin settings:
options.z_dim = 120
resnet_biggan.Generator.blocks_with_attention = "B4"
resnet_biggan.Discriminator.blocks_with_attention = "B1"

256x256
-------
Number of parameters: (D) 98,635,298 (G) 82,097,604
Required Gin settings:
options.z_dim = 140
resnet_biggan.Generator.blocks_with_attention = "B5"
resnet_biggan.Discriminator.blocks_with_attention = "B2"

512x512
-------
Number of parameters: (D)  98,801,378 (G) 82,468,068
Required Gin settings:
options.z_dim = 160
resnet_biggan.Generator.blocks_with_attention = "B4"
resnet_biggan.Discriminator.blocks_with_attention = "B3"

1024x1024
-------
Number of parameters: (D)  98,801,378 (G) 82,468,068
Required Gin settings:
options.z_dim = 180
resnet_biggan.Generator.blocks_with_attention = "B4"
resnet_biggan.Discriminator.blocks_with_attention = "B4"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from compare_gan.architectures import abstract_arch
from compare_gan.architectures import arch_ops as ops
from compare_gan.architectures import resnet_ops
from compare_gan.architectures.arch_ops import linear
from compare_gan.architectures.arch_ops import lrelu
from compare_gan.architectures.arch_ops import minibatch_stddev_layer

import numpy as np
import gin
from six.moves import range
import tensorflow as tf

class BigGanResNetBlock(resnet_ops.ResNetBlock):
  """ResNet block with options for various normalizations.

  This block uses a 1x1 convolution for the (optional) shortcut connection.
  """

  def __init__(self,
               add_shortcut,#=True,
               demodulate,#=False,
               **kwargs):
    """Constructs a new ResNet block for BigGAN.

    Args:
      add_shortcut: Whether to add a shortcut connection.
      **kwargs: Additional arguments for ResNetBlock.
    """
    super(BigGanResNetBlock, self).__init__(**kwargs)
    self._add_shortcut = add_shortcut
    self._demodulate = demodulate

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
    if inputs.shape[-1].value != self._in_channels:
      raise ValueError(
          "Unexpected number of input channels (expected {}, got {}).".format(
              self._in_channels, inputs.shape[-1].value))

    with tf.variable_scope(self._name, values=[inputs]):
      outputs = inputs

      outputs = self.batch_norm(
          outputs, z=z, y=y, is_training=is_training, name="bn1")
      if self._layer_norm:
        outputs = ops.layer_norm(outputs, is_training=is_training, scope="ln1")

      outputs = tf.nn.relu(outputs)
      if self._demodulate:
        outputs = self._style_mod_conv_layer(outputs, z, self._in_channels, self._out_channels, self._scale1, suffix="conv1")
      else:
        outputs = self._get_conv(
          outputs, self._in_channels, self._out_channels, self._scale1,
          suffix="conv1")

      outputs = self.batch_norm(
          outputs, z=z, y=y, is_training=is_training, name="bn2")
      if self._layer_norm:
        outputs = ops.layer_norm(outputs, is_training=is_training, scope="ln2")

      outputs = tf.nn.relu(outputs)
      if self._demodulate:
        outputs = self._style_mod_conv_layer(outputs, z, self._out_channels, self._out_channels, self._scale2, suffix="conv2")
      else:
        outputs = self._get_conv(
          outputs, self._out_channels, self._out_channels, self._scale2,
          suffix="conv2")

      # Combine skip-connection with the convolved part.
      if self._add_shortcut:
        shortcut = self._get_conv(
            inputs, self._in_channels, self._out_channels, self._scale,
            kernel_size=(1, 1),
            suffix="conv_shortcut")
        outputs += shortcut
      logging.info("[Block] %s (z=%s, y=%s) -> %s", inputs.shape,
                   None if z is None else z.shape,
                   None if y is None else y.shape, outputs.shape)
      return outputs


@gin.configurable
class Generator(abstract_arch.AbstractGenerator):
  """ResNet-based generator supporting resolutions 32, 64, 128, 256, 512, 1024."""

  def __init__(self,
               ch,#=96,
               blocks_with_attention,#="64",
               hierarchical_z,#=True,
               embed_z,#=False,
               embed_y,#=True,
               embed_y_dim,#=128,
               embed_bias,#=False,
               channel_multipliers,#=None,
               plain_tanh,#=False,
               use_mapping_network,#=False,
               mapping_lrmul,#=1.0,
               demodulate,#=False,
               **kwargs):
    """Constructor for BigGAN generator.

    Args:
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      hierarchical_z: Split z into chunks and only give one chunk to each.
        Each chunk will also be concatenated to y, the one hot encoded labels.
      embed_z: If True use a learnable embedding of z that is used instead.
        The embedding will have the length of z.
      embed_y: If True use a learnable embedding of y that is used instead.
      embed_y_dim: Size of the embedding of y.
      embed_bias: Use bias with for the embedding of z and y.
      **kwargs: additional arguments past on to ResNetGenerator.
    """
    super(Generator, self).__init__(**kwargs)
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._blocks_with_attention.discard('')
    self._channel_multipliers = None if channel_multipliers is None else [int(x.strip()) for x in channel_multipliers.split(",")]
    self._hierarchical_z = hierarchical_z
    self._embed_z = embed_z
    self._embed_y = embed_y
    self._embed_y_dim = embed_y_dim
    self._embed_bias = embed_bias
    self._plain_tanh = plain_tanh
    self._use_mapping_network = use_mapping_network
    self._mapping_lrmul = mapping_lrmul
    self._demodulate = demodulate

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=True,
        layer_norm=False,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm,
        add_shortcut=True,
        demodulate=self._demodulate)

  def _get_in_out_channels(self):
    resolution = self._image_shape[0]
    if self._channel_multipliers is not None:
      channel_multipliers = self._channel_multipliers
    elif resolution == 1024:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1, 1]
    elif resolution == 512:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1]
    elif resolution == 256:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1]
    elif resolution == 128:
      channel_multipliers = [16, 16, 8, 4, 2, 1]
    elif resolution == 64:
      channel_multipliers = [16, 16, 8, 4, 2]
    elif resolution == 32:
      channel_multipliers = [4, 4, 4, 4]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self._ch * c for c in channel_multipliers[:-1]]
    out_channels = [self._ch * c for c in channel_multipliers[1:]]
    return in_channels, out_channels

  def apply(self, z, y, is_training):
    """Build the generator network for the given inputs.

    Args:
      z: `Tensor` of shape [batch_size, z_dim] with latent code.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, are we in train or eval model.

    Returns:
      A tensor of size [batch_size] + self._image_shape with values in [0, 1].
    """
    z_in = z
    z = None # clear z for now
    shape_or_none = lambda t: None if t is None else t.shape
    logging.info("[Generator] inputs are z_in=%s, y=%s", z_in.shape, shape_or_none(y))
    # Each block upscales by a factor of 2.
    seed_size = 4
    z_dim = z_in.shape[1].value

    in_channels, out_channels = self._get_in_out_channels()
    num_blocks = len(in_channels)

    if self._embed_z:
      z = ops.linear(z, z_dim, scope="embed_z", use_sn=False,
                     use_bias=self._embed_bias)

    # Begin latents in -> mapping network.
    x = z_in

    if self._use_mapping_network:

      # Normalize latents.
      #if True: # if self._normalize_latents: ...
      #  x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)
        
      # Mapping layers.
      dlatent_size = z_dim
      fmaps = dlatent_size
      fan_in = z_dim * fmaps # this would need to be different if the layer sizes below were different
      gain = 1
      he_std = gain / np.sqrt(fan_in) # He init
      mapping_lrmul = self._mapping_lrmul
      init_std = 1.0 / mapping_lrmul
      runtime_coef = he_std * mapping_lrmul # Naming conventions from StyleGAN
      x = lrelu(linear(x, z_dim, lrmul=runtime_coef, scope="w_fc0", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      x = lrelu(linear(x, fmaps, lrmul=runtime_coef, scope="w_fc1", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      x = lrelu(linear(x, fmaps, lrmul=runtime_coef, scope="w_fc2", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      x = lrelu(linear(x, fmaps, lrmul=runtime_coef, scope="w_fc3", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      x = lrelu(linear(x, fmaps, lrmul=runtime_coef, scope="w_fc4", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      x = lrelu(linear(x, fmaps, lrmul=runtime_coef, scope="w_fc5", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      x = lrelu(linear(x, fmaps, lrmul=runtime_coef, scope="w_fc6", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      x = lrelu(linear(x, z_dim, lrmul=runtime_coef, scope="w_fc7", stddev=init_std, bias_start=0.0, use_sn=self._spectral_norm, use_bias=True))
      # End mapping network.

    # Warped latents z <- x
    z = x # so, z is basically 'w' from StyleGAN, the dlatent

    if self._embed_y:
      y = ops.linear(y, self._embed_y_dim, scope="embed_y", use_sn=False,
                     use_bias=self._embed_bias)
    y_per_block = num_blocks * [y]

    # Broadcast.
    if self._hierarchical_z:
      z_per_block = tf.split(z, num_blocks + 1, axis=1)
      z0, z_per_block = z_per_block[0], z_per_block[1:]
      if y is not None:
        y_per_block = [tf.concat([zi, y], 1) for zi in z_per_block]
    else:
      z0 = z
      z_per_block = num_blocks * [z]

    # Update moving average of W. 
    # (todo)
    
    # Perform style mixing regularization.
    # (todo)

    # Apply truncation trick. (Apply StyleGAN-style truncation of W?)
    # (todo)

    logging.info("[Generator] z0=%s, z_per_block=%s, y_per_block=%s",
                 z0.shape, [str(shape_or_none(t)) for t in z_per_block],
                 [str(shape_or_none(t)) for t in y_per_block])

    # Map noise to the actual seed.
    net = ops.linear(
        z0,
        in_channels[0] * seed_size * seed_size,
        scope="fc_noise",
        use_sn=self._spectral_norm)
    # Reshape the seed to be a rank-4 Tensor.
    net = tf.reshape(
        net,
        [-1, seed_size, seed_size, in_channels[0]],
        name="fc_reshaped")

    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "B{}".format(block_idx + 1)
      block = self._resnet_block(
          name=name,
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale="up")
      net = block(
          net,
          z=z_per_block[block_idx],
          y=y_per_block[block_idx],
          is_training=is_training)
      res = net.shape[1].value
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        logging.info("[Generator] Applying non-local block at %dx%d resolution to %s",
                     res, res, net.shape)
        net = ops.non_local_block(net, "non_local_block",
                                  use_sn=self._spectral_norm)
    assert len(blocks_with_attention) <= 0

    # Final processing of the net.
    # Use unconditional batch norm.
    logging.info("[Generator] before final processing: %s", net.shape)
    net = ops.batch_norm(net, is_training=is_training, name="final_norm")
    net = tf.nn.relu(net)
    net = ops.conv2d(net, output_dim=self._image_shape[2], k_h=3, k_w=3,
                     d_h=1, d_w=1, name="final_conv",
                     use_sn=self._spectral_norm)
    logging.info("[Generator] after final processing: %s", net.shape)
    if self._plain_tanh:
      net = tf.nn.tanh(net)
    else:
      net = (tf.nn.tanh(net) + 1.0) / 2.0
    return net


@gin.configurable
class Discriminator(abstract_arch.AbstractDiscriminator):
  """ResNet-based U-net discriminator supporting resolutions 32, 64, 128, 256, 512, 1024."""

  def __init__(self,
               ch,#=96,
               blocks_with_attention,#="64",
               project_y,#=True,
               channel_multipliers,#=None,
               mbstddev,#=False,
               mbstddev_group_size,#=4, # 4 from StyleGAN
               mbstddev_new_features,#=1, # 1 from StyleGAN
               **kwargs):
    """Constructor for BigGAN U-net discriminator.

    Args:
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      project_y: Add an embedding of y in the output layer.
      **kwargs: additional arguments past on to ResNetDiscriminator.
    """
    super(Discriminator, self).__init__(**kwargs)
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._blocks_with_attention.discard('')
    self._channel_multipliers = None if channel_multipliers is None else [int(x.strip()) for x in channel_multipliers.split(",")]
    self._project_y = project_y
    self._mbstddev = mbstddev
    self._mbstddev_group_size = mbstddev_group_size
    self._mbstddev_new_features = mbstddev_new_features

  def _resnet_block_encode_down(self, name, in_channels, out_channels, scale):
    """ResNet block for the encoder discriminator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown enc discriminator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=False,
        layer_norm=self._layer_norm,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm,
        add_shortcut=in_channels != out_channels,
        demodulate=False)

  def _resnet_block_decode_up(self, name, in_channels, out_channels, scale):
    """ResNet block for the decoder discriminator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown dec discriminator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=True, # this variable is incorrectly named, it is only checked to apply 'scale' first or second  
        layer_norm=False,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm,
        add_shortcut=in_channels != out_channels,
        demodulate=False)

  def _get_in_out_channels_encode_down(self, colors, resolution):
    in_channels, out_channels = [], []
    if colors not in [1, 3]:
      raise ValueError("Unsupported color channels: {}".format(colors))
    if self._channel_multipliers is not None:
      channel_multipliers = self._channel_multipliers
    # elif resolution == 1024:
    #   channel_multipliers = [1, 1, 1, 2, 4, 8, 8, 16, 16]
    # elif resolution == 512:
    #   channel_multipliers = [1, 1, 2, 4, 8, 8, 16, 16]
    # elif resolution == 256:
    #   channel_multipliers = [1, 2, 4, 8, 8, 16, 16]
    elif resolution == 128:
      in_channel_multipliers =  [1, 2, 4, 8, 16] # note: omit first index due to constant color channels 
      out_channel_multipliers = [1, 2, 4, 8, 16, 16]
      in_channels =  [colors] + [self._ch * c for c in in_channel_multipliers]
      out_channels = [self._ch * c for c in out_channel_multipliers]
    # elif resolution == 64:
    #   channel_multipliers = [2, 4, 8, 16, 16]
    # elif resolution == 32:
    #   channel_multipliers = [2, 2, 2, 2]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    assert len(in_channels) > 0
    assert len(out_channels) > 0
    return in_channels, out_channels

  def _get_in_out_channels_decode_up(self, image_resolution):
    resolution = image_resolution
    in_channels, out_channels = [], []
    if self._channel_multipliers is not None:
      channel_multipliers = self._channel_multipliers
    # elif resolution == 1024:
    #   channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1, 1]
    # elif resolution == 512:
    #   channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1]
    # elif resolution == 256:
    #   channel_multipliers = [16, 16, 8, 8, 4, 2, 1]
    elif resolution == 128:
      in_channel_multipliers =  [32, 16, 8, 4, 2] # this differs from the U-net disc paper, since we include the resblock that doesn't perform a downscale on ch
      out_channel_multipliers = [ 8,  4, 2, 1, 1]
      in_channels = [self._ch * c for c in in_channel_multipliers]
      out_channels = [self._ch * c for c in out_channel_multipliers]
    # elif resolution == 64:
    #   channel_multipliers = [16, 16, 8, 4, 2]
    # elif resolution == 32:
    #   channel_multipliers = [4, 4, 4, 4]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    assert len(in_channels) > 0
    assert len(out_channels) > 0
    return in_channels, out_channels


  def apply(self, x, y, is_training):
    """Apply the discriminator on a input.

    Args:
      x: `Tensor` of shape [batch_size, ?, ?, ?] with real or fake images.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: Boolean, whether the architecture should be constructed for
        training or inference.

    Returns:
      Tuple of 3 Tensors, the final prediction of the discriminator, the logits
      before the final output activation function and logits form the second
      last layer.
    """
    logging.info("[DiscriminatorEnc] inputs are x=%s, y=%s", x.shape,
                 None if y is None else y.shape)
    resnet_ops.validate_image_inputs(x)

    # Encoder (down) part of the U-net discriminator
    in_channels, out_channels = self._get_in_out_channels_encode_down(
        colors=x.shape[-1].value, resolution=x.shape[1].value)
    num_blocks = len(in_channels)
    block_list = []
    net = x
    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "enc_B{}".format(block_idx + 1)
      is_last_block = block_idx == num_blocks - 1
      extra_channels = 0
      if self._mbstddev and is_last_block:
        group_size = self._mbstddev_group_size # 4 from StyleGAN
        num_new_features = self._mbstddev_new_features # 1 from StyleGAN
        net = minibatch_stddev_layer(net, group_size, num_new_features)
        extra_channels += num_new_features
      logging.info("[DiscriminatorEnc] Block %s at %dx%d resolution: %s", name, net.shape[1].value, net.shape[1].value, net.shape)
      block = self._resnet_block_encode_down(
          name=name,
          in_channels=in_channels[block_idx]+extra_channels,
          out_channels=out_channels[block_idx],
          scale="none" if is_last_block else "down")
      net = block(net, z=None, y=y, is_training=is_training)
      res = net.shape[1].value
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        logging.info("[DiscriminatorEnc] Applying non-local block at %dx%d resolution to %s",
                     res, res, net.shape)
        net = ops.non_local_block(net, "non_local_block",
                                  use_sn=self._spectral_norm)
      if not is_last_block:
        block_list.append(net)
    assert len(blocks_with_attention) <= 0

    # Preserve this to compute the standard loss for discriminator
    to_decoder = net
    
    # Final part of the DiscEnc
    logging.info("[DiscriminatorEnc] before final processing: %s", net.shape)
    net = tf.nn.relu(net)
    h = tf.math.reduce_sum(net, axis=[1, 2])
    out_logit = ops.linear(h, 1, scope="final_fc", use_sn=self._spectral_norm)
    logging.info("[DiscriminatorEnc] after final processing: %s", net.shape)
    if self._project_y:
      if y is None:
        raise ValueError("You must provide class information y to project.")
      with tf.variable_scope("embedding_fc"):
        y_embedding_dim = out_channels[-1]
        # We do not use ops.linear() below since it does not have an option to
        # override the initializer.
        kernel = tf.get_variable(
            "kernel", [y.shape[1], y_embedding_dim], tf.float32,
            initializer=tf.initializers.glorot_normal())
        kernel = ops.graph_spectral_norm(kernel)
        if self._spectral_norm:
          kernel, norm = ops.spectral_norm(kernel)
        embedded_y = tf.matmul(y, kernel)
        logging.info("[DiscriminatorEnc] embedded_y for projection: %s",
                     embedded_y.shape)
        out_logit += tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
    out = tf.nn.sigmoid(out_logit)

    # Decoder (up) part of the U-net discriminator
    block_list.reverse()
    net = to_decoder
    logging.info("[DiscriminatorEnc] block_list: %s", [str(x.name)+str(x.shape) for x in block_list])
    logging.info("[DiscriminatorDec] begin decoder: %s", net.shape)
    image_resolution = int(x.shape[1])
    in_channels, out_channels = self._get_in_out_channels_decode_up(image_resolution)
    num_blocks = len(in_channels)
    logging.info("[DiscriminatorDec] # blocks, in, out: %s %s %s", num_blocks, in_channels, out_channels)
    for block_idx in range(num_blocks):
      name = "dec_B{}".format(block_idx + 1)
      is_last_block = block_idx == num_blocks - 1
      partner = block_list[block_idx]
      logging.info("[DiscriminatorDec] %s, block_idx %s", name, block_idx)
      logging.info("[DiscriminatorDec] concat from_enc: %s %s", partner.name, partner.shape)
      logging.info("[DiscriminatorDec] concat into_dec: %s %s", net.name, net.shape)
      net = tf.concat([net, partner], axis=3)
      logging.info("[DiscriminatorDec] Block %s at %dx%d resolution: %s", name, net.shape[1].value, net.shape[1].value, net.shape)
      block = self._resnet_block_decode_up(
          name=name,
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale="up")
      net = block(net, z=None, y=y, is_training=is_training)
    
    # Final part of the DiscDec
    logging.info("[DiscriminatorDec] before final processing: %s", net.shape)
    net = tf.nn.relu(net)
    # U-net Disc_Dec modifications: use "out channels" instead of 3 (RGB channels)
    modified_final_output_channels = out_channels[-1]
    net = ops.conv2d(net, output_dim=modified_final_output_channels,
                     k_h=3, k_w=3,
                     d_h=1, d_w=1, name="final_conv3x3",
                     use_sn=self._spectral_norm)
    # U-net Disc_Dec modifications: use a conv1x1 to collapse to grayscale
    net = ops.conv2d(net, output_dim=1, k_h=1, k_w=1,
                     d_h=1, d_w=1, name="final_conv1x1",
                     use_sn=self._spectral_norm)

    mask_logit = net
    mask = tf.nn.sigmoid(mask_logit) # from the U-net GAN discriminator design
    logging.info("[DiscriminatorDec] after final processing: %s", mask.shape)
    logging.info("[Discriminator] FINAL OUT %s %s %s %s %s", out, out_logit, h, mask, mask_logit)

    # return out, out_logit, h
    return out, out_logit, (h, mask, mask_logit)
