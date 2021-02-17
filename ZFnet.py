

def ZFnet(tensor, nb_classes, activation='relu'):
  from keras.models import Model
  from keras import layers
  from . import conv2d
  Inputs = layers.Input(shape=(tensor)+ (3, ))
  x= conv2d(Inputs, 96, 7, 2, 'same', True, activation= activation)
  x= layers.MaxPooling2D(3, strides=2, padding='valid')(x)
  x= conv2d(x, 256, 5, 2, 'same', True, activation=activation)
  x= layers.MaxPooling2D(3, strides=2, padding='valid')(x)
  x= conv2d(x, 384, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 384, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 256, 3, 1, 'same', True, activation= activation)
  x= layers.MaxPooling2D(3, strides=2, padding='valid')(x)
  x= layers.Flatten()(x)
  x= layers.Dense(4096, activation= activation)(x)
  x= layers.Dropout(0.5)(x)
  x= layers.Dense(4096, activation=activation)(x)
  x= layers.Dropout(0.5)(x)
  x= layers.Dense(nb_classes, activation='softmax')(x)
  model = Model(Inputs, x)
  return model
