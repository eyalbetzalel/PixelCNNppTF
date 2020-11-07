# Build a small Pixel CNN++ model to train on MNIST.

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import matplotlib.pyplot as plt


tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto(log_device_placement=True)
sess = tf.compat.v1.Session(config=config)
tf.debugging.set_log_device_placement(True)


######### Sample ##############

def imsave(batch_or_tensor, title=None, figsize=None, filename="sample.png"):
  """Renders tensors as an image using Matplotlib.
  Args:
    batch_or_tensor: A batch or single tensor to render as images. If the batch
      size > 1, the tensors are flattened into a horizontal strip before being
      rendered.
    title: The title for the rendered image. Passed to Matplotlib.
    figsize: The size (in inches) for the image. Passed to Matplotlib.
  """
  batch = batch_or_tensor
  for _ in range(batch.ndim):
    batch = batch.unsqueeze(0)
  n, c, h, w = batch.shape
  tensor = batch.permute(1, 2, 0, 3).reshape(c, h, -1)

  plt.figure(figsize=figsize)
  plt.title(title)
  plt.axis('off')
  plt.imsave(filename,image)
 
##################################
  

# Load MNIST from tensorflow_datasets
data = tfds.load('mnist')
train_data, test_data = data['train'], data['test']

def image_preprocess(x):
  x['image'] = tf.cast(x['image'], tf.float32)
  return (x['image'],)  # (input, output) of the model

batch_size = 16
train_it = train_data.map(image_preprocess).batch(batch_size).shuffle(1000)

image_shape = (28, 28, 1)
# Define a Pixel CNN network
dist = tfd.PixelCNN(
    image_shape=image_shape,
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=.3,
)

# Define the model input
image_input = tfkl.Input(shape=image_shape)

# Define the log likelihood for the loss fn
log_prob = dist.log_prob(image_input)

# Define the model
model = tfk.Model(inputs=image_input, outputs=log_prob)
model.add_loss(-tf.reduce_mean(log_prob))

# Compile and train the model
model.compile(
    optimizer=tfk.optimizers.Adam(.001),
    metrics=[])

model.fit(train_it, epochs=1, verbose=True)

# sample five images from the trained model
samples = dist.sample(5)
imsave(samples)

import ipdb
ipdb.set_trace()
