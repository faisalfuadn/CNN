

def se_block(tensor, ratio=8, activation='relu'):
    from keras import layers
    num_filters = tensor.shape[-1]
    se_shape = (1, 1, num_filters)
    reduced_channels = num_filters // ratio
    #Squeeze
    se = layers.GlobalAveragePooling2D()(tensor)
    se = layers.Reshape(se_shape)(se)
    x = layers.Dense(reduced_channels)(se)
    x = layers.core.Activation(activation=activation)(x)
    #Excitation
    x = layers.Dense(num_filters, activation='sigmoid')(x)
    
    x = layers.multiply([x, tensor])
    return x
