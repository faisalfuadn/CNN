# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:19:24 2021

@author: suantara
"""

from . import conv2d
from keras import layers
import keras.backend as K

def inception_resnetA(tensor,scale, activation='relu'):
    pad = 'same'
    a = conv2d(tensor, 32, 1, 1, pad, True, activation= activation)
    b = conv2d(tensor, 32, 1, 1, pad, True, activation= activation)
    b = conv2d(b, 32, 3, 1, pad, True, activation= activation)
    c = conv2d(tensor, 32, 1, 1, pad, True, activation= activation)
    c = conv2d(c, 48, 3, 1, pad, True, activation= activation)
    c = conv2d(c, 64, 3, 1, pad, True, activation= activation)
    x = layers.concatenate([a, b, c], axis=-1)
    conv1x1 = conv2d(x, 384, 1, 1, pad, False)
    final_conv = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(tensor)[1:],
                      arguments={'scale': scale})([tensor, conv1x1])
    return final_conv

def inception_resnetB(tensor, scale, activation='relu'):
    pad = 'same'
    a = conv2d(tensor, 192, 1, 1, pad, True, activation= activation)
    b = conv2d(tensor, 128, 1, 1, pad, True, activation= activation)
    b = conv2d(b, 160, [1,7], 1, pad, True, activation= activation)
    b = conv2d(b, 192, [7,1], 1, pad, True, activation= activation)
    x = layers.concatenate([a, b], axis=-1)
    conv1x1 = conv2d(x, 1152, 1, 1, pad, False)
    final_conv = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(tensor)[1:],
                      arguments={'scale': scale})([tensor, conv1x1])
    return final_conv

def inception_resnetC(tensor, scale, activation='relu'):
    pad = 'same'
    a = conv2d(tensor,192,1,1,pad,True, activation= activation)
    b = conv2d(tensor,192,1,1,pad,True,activation= activation)
    b = conv2d(b,224,[1,3],1,pad,True, activation= activation)
    b = conv2d(b,256,[3,1],1,pad,True, activation= activation)
    x = layers.concatenate([a, b], axis=-1)
    conv1x1 = conv2d(x, 2048, 1, 1, pad, False)
    final_conv = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(tensor)[1:],
                      arguments={'scale': scale})([tensor, conv1x1])
    return final_conv

def Stem_block(tensor):
  x = conv2d(tensor, 32, 3, 2, 'valid', True)
  x = conv2d(x, 32, 3, 1, 'valid', True)
  x = conv2d(x, 64, 3, 1, 'valid', True)

  a = layers.MaxPooling2D(3, strides=1, padding='valid')(x)
  b = conv2d(x, 64, 3, 1, 'valid', True)

  x = layers.concatenate([a, b], axis=-1)

  a = conv2d(x, 64, 1, 1, 'same', True)
  a = conv2d(a, 64, [1,7], 1, 'same', True)
  a = conv2d(a, 64, [7,1], 1, 'same', True)
  a = conv2d(a, 96, 3, 1, 'valid', True)

  b = conv2d(x, 64, 1, 1, 'same', True)
  b = conv2d(b, 96, 3, 1, 'valid', True)

  x = layers.concatenate([a, b], axis=-1)

  a = conv2d(x, 192, 3, 1, 'valid', True)
  b = layers.MaxPooling2D(3, strides=1, padding='valid')(x)
  x = layers.concatenate([a, b], axis=-1)
  return x

def reductionA(tensor, activation='relu'):
  #35 × 35 to 17 × 17 reduction module.
  a = layers.MaxPooling2D(3, strides=2, padding='valid')(tensor)

  b = conv2d(tensor, 384, 3, 2, 'valid', True, activation= activation)

  c = conv2d(tensor,256, 1, 1,'same',True, activation= activation)
  c = conv2d(c, 256, 3, 1, 'same', True, activation= activation)
  c = conv2d(c, 384, 3, 2, 'valid', True, activation= activation)

  x = layers.concatenate([a, b, c], axis=-1)
  return x

def reductionB(tensor, activation='relu'):
  #17 × 17 to 8 × 8 reduction module.
  a = layers.MaxPooling2D(3, strides=2, padding='valid')(tensor)

  b = conv2d(tensor, 256, 1, 1, 'same', True, activation= activation)
  b = conv2d(b, 384, 3, 2, 'valid', True, activation= activation)

  c = conv2d(tensor, 256, 1, 1, 'same', True, activation= activation)
  c = conv2d(c, 256, 3, 2, 'valid', True, activation= activation)

  d = conv2d(tensor, 256, 1, 1, 'same', True, activation= activation)
  d = conv2d(d, 256, 3, 1, 'same', True, activation= activation)
  d = conv2d(d, 256, 3, 2, 'valid', True, activation= activation)

  x = layers.concatenate([a, b, c, d], axis=-1)
  return x

def inception_resnetv2(img_size, block_configuration, n_labels, activation, att=False, attention=None):
  from . import se_block
  from . import cbam_block
  from keras.models import Model
  Inputs = layers.Input(shape=(img_size)+ (3, ))
  #stem block
  x= Stem_block (Inputs)

  #inception resnet A
  for i in range (block_configuration[0]):
      x= inception_resnetA(x, 0.17, activation=activation)
  x = reductionA (x, activation=activation) #reduction A
  if att==True:
    if attention == 'SE':
      x = se_block(x, activation= activation)
    elif attention =='CBAM':
      x = cbam_block(x, activation= activation)

  #inception resnet B
  for i in range (block_configuration[1]):
      x= inception_resnetB(x,0.1, activation=activation)
  x = reductionB(x) #reduction B
  if att==True:
    if attention == 'SE':
      x = se_block(x, activation= activation)
    elif attention =='CBAM':
      x = cbam_block(x, activation= activation)

  #Inception Resnet C
  for i in range (block_configuration[2]):
      x= inception_resnetC(x,0.2, activation=activation)
  if att==True:
    if attention == 'SE':
      x = se_block(x, activation= activation)
    elif attention =='CBAM':
      x = cbam_block(x, activation= activation)

  #TOP
  x = layers.GlobalAveragePooling2D(data_format='channels_last')(x)
  x = layers.Dropout(0.6)(x)
  x = layers.Dense(n_labels, activation='softmax')(x)
  model = Model(Inputs, x)
  return model
