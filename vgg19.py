

def VGG19(tensor, n_labels, activation='relu'):
  from keras.models import Model
  from keras import layers
  from . import conv2d
  
  Inputs= layers.Input(shape=(tensor) + (3,))
  x= conv2d(Inputs, 64, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 64, 3, 1, 'same', True, activation= activation)
  x= layers.MaxPooling2D(2, strides=2)(x)

  x= conv2d(x, 128, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 128, 3, 1, 'same', True, activation= activation)
  x= layers.MaxPooling2D(2, strides=2)(x)

  x= conv2d(x, 256, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 256, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 256, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 256, 3, 1, 'same', True, activation= activation)
  x= layers.MaxPooling2D(2, strides=2)(x)

  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= layers.MaxPooling2D(2, strides=2)(x)

  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= conv2d(x, 512, 3, 1, 'same', True, activation= activation)
  x= layers.MaxPooling2D(2, strides=2)(x)

  x= layers.Flatten()(x)
  x= layers.Dense(4096, activation= activation)(x)
  x= layers.Dropout(0.5)(x)
  x= layers.Dense(4096, activation=activation)(x)
  x= layers.Dropout(0.5)(x)
  x= layers.Dense(n_labels, activation='softmax')(x)
  model = Model(Inputs, x)
  return model
