
#import tensorflow as tf

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()


dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))

dataset = dataset.shuffle(1000).batch(32)

mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu', # convoutional layer
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'), # convoutional layer
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10) # output layer
])

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []


def train_step(images, labels):
  with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)

    # Add asserts to check the shape of the output.
    tf.debugging.assert_equal(logits.shape, (32, 10))

    loss_value = loss_object(labels, logits)

  loss_history.append(loss_value.numpy().mean())
  grads = tape.gradient(loss_value, mnist_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
  return np.mean(loss_history)


def train(epochs):
  tot_loss = []
  for epoch in range(epochs):
    for (batch, (images, labels)) in enumerate(dataset):
      b_loss = train_step(images, labels)
      tot_loss.append(b_loss)
    #print ('Epoch {} finished'.format(epoch))
  return np.mean(tot_loss)
    
final_loss = train(epochs = 1)
print(final_loss)
