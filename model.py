import tensorflow as tf

from tensorflow.keras import Model

from module import Controller, Memory


class DNC(Model):

    def __init__(self, opts):

        super(DNC, self).__init__()
        self.controller = Controller(opts)
        self.memory = Memory(opts)

    def __call__(self, src, state):

        ctrl_state = state['ctrl']
        memory_state = state['memory']

        # Concat input and initial read vectors.
        batch_size = src.shape[0]
        read_heads = self.memory.read_heads
        addr_size = self.memory.address_size
        flat_dim = read_heads * addr_size
        read_vectors = tf.reshape(
            memory_state['read_vec'],
            [batch_size, flat_dim]
        )

        # Run controller.
        ctrl_input = tf.concat([src, read_vectors], axis=1)
        ctrl_output, ctrl_state = self.controller(
            ctrl_input,
            ctrl_state
        )

        # Access memory.
        memory_output, memory_state = self.memory(
            ctrl_output['interface'],
            memory_state
        )

        output = ctrl_output['output'] + memory_output
        output = tf.math.sigmoid(output)

        state = {
            'ctrl': ctrl_state,
            'memory': memory_state
        }

        return (output, state)
