import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def gen_binseq_batch(batch_size, seq_len, bins):

    shape = (batch_size, seq_len)
    source = np.zeros(shape, dtype=np.int64)
    target = np.zeros(batch_size, dtype=np.int64)
    max_val = int('1' * seq_len, base=2)
    bin_size = (max_val + 1) / bins

    for i in range(batch_size):
        sample = np.random.randint(2, size=(seq_len,))
        binstr = ''.join(str(i) for i in sample)
        target[i] = (int(binstr, base=2) / bin_size)
        source[i, :] = sample

    src_batch = tf.constant(source)
    tgt_batch = tf.constant(target)
    return (src_batch, tgt_batch)


def gen_copytask_batch(batch_size, n_pairs, seq_len, vec_dim):

    src_batch = []
    tgt_batch = []
    for _ in range(n_pairs):
        x, y = gen_copypairs_batch(
            batch_size,
            seq_len,
            vec_dim
        )

        src_batch.append(x)
        tgt_batch.append(y)

    src_batch = np.concatenate(src_batch, axis=1)
    tgt_batch = np.concatenate(tgt_batch, axis=1)
    return (src_batch, tgt_batch)


def gen_copypairs_batch(batch_size, seq_len, vec_dim):

    total_len = 2 * seq_len + 2
    shape = (batch_size, total_len, vec_dim)
    inp_seq = np.zeros(shape, dtype=np.float32)
    out_seq = np.zeros(shape, dtype=np.float32)

    for i in range(batch_size):
        sample_shape = (seq_len, vec_dim-1)
        sample = np.random.randint(2, size=sample_shape)
        out_seq[i, seq_len+1:2*seq_len+1, :-1] = sample
        inp_seq[i, :seq_len, :-1] = sample
        inp_seq[i, seq_len, vec_dim-1] = 1

    return (inp_seq, out_seq)


if __name__ == '__main__':

    src, tgt = gen_copytask_batch(32, 3, 5, 8)
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(src[0].T)
    ax[1].imshow(tgt[0].T)
    plt.show()
