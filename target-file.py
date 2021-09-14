import numpy as np
import tensorflow as tf


def train_step(images, labels, mnist_model, optimizer, loss_object):
    loss_history = []
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        tf.debugging.assert_equal(logits.shape, (32, 10))
        loss_value = loss_object(labels, logits)
    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return np.mean(loss_history), mnist_model


def train(epochs, dataset, train_step):
    tot_loss = []
    weights = None

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            b_loss, mnist_model = train_step(images, labels, mnist_model, optimizer, loss_object)
            tot_loss.append(b_loss)
    return mnist_model


def rev_value():
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32), tf.cast(mnist_labels,tf.int64)))
    dataset = dataset.shuffle(1000).batch(32)
    return train(1, dataset, train_step)


model = rev_value()