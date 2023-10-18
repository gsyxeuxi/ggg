import tensorflow as tf

assert tf.config.list_physical_devices('GPU')
assert tf.test.is_built_with_cuda()