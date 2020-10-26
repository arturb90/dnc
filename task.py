import numpy as np


class Task:

    generate = {
        'copyrepeat': (lambda x, y: Task.generate_copyrepeat(x, y))
    }

    @staticmethod
    def generate_batch(task_opts, batch_size):

        return Task.generate[task_opts.task](
            task_opts,
            batch_size
        )

    @staticmethod
    def generate_copyrepeat(task_opts, batch_size):

        src_batch = []
        tgt_batch = []
        for _ in range(task_opts.n_pairs):
            x, y = Task.generate_copypairs(
                batch_size,
                task_opts.seq_len,
                task_opts.vec_len
            )

            src_batch.append(x)
            tgt_batch.append(y)

        src_batch = np.concatenate(src_batch, axis=1)
        tgt_batch = np.concatenate(tgt_batch, axis=1)
        return (src_batch, tgt_batch)

    @staticmethod
    def generate_copypairs(batch_size, seq_len, vec_dim):

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
