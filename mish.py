

from keras import backend as K
def mish(inputs):
  return inputs * K.tanh(K.softplus(inputs))
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'mish': mish})
