# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:26:52 2021

@author: suantara
"""


def cbam_block(tensor, ratio=8, activation='relu'):
    from keras import layers
    import keras.backend as K
    channel = tensor.shape[-1]
	
    avg_pool = layers.GlobalAveragePooling2D()(tensor)    
    avg_pool = layers.Reshape((1,1,channel))(avg_pool)
    avg_pool = layers.Dense(channel//ratio, activation= activation)(avg_pool)
    avg_pool = layers.Dense(channel, kernel_initializer='he_normal')(avg_pool)
  
    max_pool = layers.GlobalMaxPooling2D()(tensor)
    max_pool = layers.Reshape((1,1,channel))(max_pool)
    max_pool = layers.Dense(channel//ratio, activation= activation)(max_pool)
    max_pool = layers.Dense(channel, kernel_initializer='he_normal')(max_pool)
  
    channel_att = layers.add([avg_pool,max_pool])
    channel_att = layers.core.Activation('sigmoid')(channel_att)
    channel_att= layers.multiply([tensor, channel_att])
  
    #spatial attention
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_att)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_att)
    concat = layers.concatenate([avg_pool, max_pool], axis= -1)
    spatial_att = layers.Conv2D(filters = 1,
            kernel_size=7,
            strides=1,
            padding='same',
            activation='sigmoid')(concat)	
      
    return layers.multiply([channel_att, spatial_att])