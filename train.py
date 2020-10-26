import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DEBUG = False

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from model import DNC
from opts import get_opts
from task import Task


def train(opts, model):

    task_opts = opts['task']
    sess_opts = opts['session']

    # Optimizer and criterion.
    opt = optimizers.SGD(learning_rate=0.01)
    crit = losses.CategoricalCrossentropy(from_logits=True)

    # Accuracy metric.
    train_acc = metrics.CategoricalAccuracy(name='train_acc')

    # Generate batch for selected task.
    src, tgt = Task.generate_batch(
        task_opts,
        sess_opts.batch_size
    )

    for epoch in range(sess_opts.epochs):
        train_acc.reset_states()

        p, loss = train_step(model, src, tgt, crit, opt)
        train_acc(tgt, p)

        if epoch % 20 == 0:
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(src[0].T)
            ax[1].imshow(p.numpy()[0].T)
            plt.show()

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {loss}, '
            f'Accuracy: {train_acc.result() * 100} '
        )


# @tf.function(autograph=not DEBUG)
def train_step(model, src, tgt, crit, opt):

    with tf.GradientTape() as tape:

        loss = 0
        state = {
            'ctrl': model.controller.initialize(src.shape[0]),
            'memory': model.memory.initialize(src.shape[0])
        }

        temp = tf.zeros(tgt.shape)
        predictions = tf.Variable(temp)
        for i in range(src.shape[1]):
            output, state = model(src[:, i, :], state)
            loss += crit(tgt[:, i, :], output)
            predictions[:, i, :].assign(output)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return predictions, loss.numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Session options.
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)

    # Model options.
    parser.add_argument('--controller', type=str, default='lstm')
    parser.add_argument('--memory_size', type=int, default=16)
    parser.add_argument('--address_size', type=int, default=8)
    parser.add_argument('--read_heads', type=int, default=2)
    parser.add_argument('--lstm_units', type=int, default=1024)
    parser.add_argument('--hid_sizes', type=int, nargs='+', default=[])

    # Task options
    parser.add_argument('--task', type=str, default='copyrepeat')
    parser.add_argument('--seq_len', type=int, default=1)
    parser.add_argument('--vec_len', type=int, default=2)
    parser.add_argument('--n_pairs', type=int, default=1)

    opts = get_opts(parser.parse_args())
    model = DNC(opts['model'])
    train(opts, model)
