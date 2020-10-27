import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTMCell

# Minimum value ensures numerical stability.
_EPSILON = 1e-6


class Controller(Model):

    mtype = {
        'ffn': (lambda opts: FFN(opts)),
        'lstm': (lambda opts: LSTM(opts))
    }

    def __init__(self, opts):
        super(Controller, self).__init__()
        mtype = Controller.mtype[opts.controller]
        self.model = mtype(opts)

    def __call__(self, inputs, state):
        '''
        :param inputs:  Controller input vector, defined as
                        the concatenation of the input tokens
                        vector-valued embedding and the read-
                        heads from the previous time step.

        :param state:   The controller state from the previous
                        time step. Only relevant for controller
                        of type LSTM.
        '''
        return self.model(inputs, state)

    def initialize(self, batch_size):
        return self.model.initialize(batch_size)


class Memory(Model):

    def __init__(self, opts):

        super(Memory, self).__init__()
        self.memory_size = opts.memory_size
        self.address_size = opts.address_size
        self.read_heads = opts.read_heads
        self.output_size = opts.output_size

        # Interface component sizes.
        self.if_sizes = [
            # Write key, write vector, erase vector of size <address_size>.
            *[self.address_size for _ in range(3)],
            # <read_head> free_gates/read strengths.
            # Write strength, write gate, alloc gate.
            self.read_heads, self.read_heads, 3,
            # <read_head> read keys of size <address_size>
            *[self.address_size for _ in range(self.read_heads)],
            # <read_head> read modes of size 3.
            *[3 for _ in range(self.read_heads)]
        ]

        self.out_layer = Dense(
            opts.output_size,
            activation=None
        )

    def __call__(self, interface, memory_state):

        # Split the interface components.
        components = self.__split_if(interface)

        # Dynamic memory allocation.
        memory_retention = self.__memory_retention(
            components['free_gate'],
            memory_state['read_weights']
        )

        memory_usage = self.__memory_usage(
            memory_state['usage_vec'],
            memory_state['write_weights'],
            memory_retention
        )

        alloc_w = self.__alloc_weighting(memory_usage)

        # Content write weighting.
        write_addressing = self.__content_addressing(
            memory_state['memory'],
            components['write_key'],
            components['write_strength']
        )

        write_w = self.__write_interpolation(
            {'write': components['write_gate'],
             'alloc': components['alloc_gate']},
            write_addressing,
            alloc_w
        )

        memory_state['temporal_link'].update(
            memory_state['precedence_weight'],
            write_w
        )

        memory_output = self.out_layer(interface)
        return (memory_output, memory_state)

    def initialize(self, batch_size):

        memory_state = {}
        memory_state['memory'] = tf.zeros((
            batch_size,
            self.memory_size,
            self.address_size
        ))

        memory_state['read_weights'] = tf.zeros((
            batch_size,
            self.memory_size,
            self.read_heads
        ))

        memory_state['write_weights'] = tf.zeros((
            batch_size,
            self.memory_size
        ))

        memory_state['read_vec'] = tf.zeros((
            batch_size,
            self.read_heads,
            self.address_size
        ))

        memory_state['usage_vec'] = tf.zeros((
            batch_size,
            self.memory_size
        ))

        memory_state['precedence_weight'] = tf.zeros((
            batch_size,
            self.memory_size
        ))

        memory_state['temporal_link'] = TemporalLink(
            batch_size,
            self.memory_size
        )

        return memory_state

    def __memory_retention(self, free_gates, prev_read_w):

        free_gates = tf.expand_dims(free_gates, axis=1)
        temp = tf.multiply(free_gates, prev_read_w)
        temp = tf.ones(prev_read_w.shape) - temp
        memory_retention = tf.reduce_prod(temp, axis=2)
        return memory_retention

    def __memory_usage(self, prev_usage, prev_write_w, retention):

        u_add = tf.add(prev_usage, prev_write_w)
        u_muliply = tf.multiply(prev_usage, prev_write_w)
        usage = tf.multiply(tf.subtract(u_add, u_muliply), retention)
        return usage

    def __alloc_weighting(self, usage):

        indices = tf.argsort(usage)
        usage_sorted = tf.gather(usage, indices, axis=1, batch_dims=1)
        temp = tf.subtract(tf.ones(usage.shape), usage_sorted)
        cumprod = tf.math.cumprod(usage_sorted, axis=1)
        alloc_w = tf.multiply(temp, cumprod)

        # The allocation weights are sorted in
        # ascending order, sorting has to be reverted.
        fn = (lambda t: tf.math.invert_permutation(t))
        indices_inv = tf.map_fn(fn=fn, elems=indices)
        alloc_w = tf.gather(alloc_w, indices_inv, axis=1, batch_dims=1)

        return alloc_w

    def __write_interpolation(self, gate, write_addr, alloc_w):

        write_gate = tf.expand_dims(gate['write'], axis=-1)
        alloc_gate = tf.expand_dims(gate['alloc'], axis=-1)

        ones = tf.ones(alloc_gate.shape)
        inv_alloc_gate = tf.subtract(ones, alloc_gate)
        gated_write_addr = tf.multiply(inv_alloc_gate, write_addr)
        gated_alloc_w = tf.multiply(alloc_gate, alloc_w)
        gated_add = tf.add(gated_write_addr, gated_alloc_w)
        write_w = tf.multiply(write_gate, gated_add)

        return write_w

    def __content_addressing(self, memory, key, strength):

        similarity = self.__cosine_similarity(memory, key)
        strength = tf.reshape(strength, shape=(key.shape[0], 1, 1))
        similarity = tf.multiply(similarity, strength)
        similarity = tf.squeeze(similarity, axis=2)
        return tf.math.softmax(similarity, axis=1)

    def __cosine_similarity(self, memory, key):

        # Calculate denominator
        key = tf.expand_dims(key, axis=1)
        key_norm = self.__l2_norm(key, axis=2)
        memory_norms = self.__l2_norm(memory, axis=2)
        norm = tf.multiply(memory_norms, key_norm)

        # Calculate numerator.
        dot = tf.reduce_sum(memory * key, axis=2, keepdims=True)

        return dot / (norm + _EPSILON)

    def __l2_norm(self, t, axis=0):

        squared = tf.reduce_sum(t * t, axis=axis, keepdims=True)
        return tf.sqrt(squared)

    def __oneplus(self, x):

        return 1 + tf.math.softplus(x)

    def __split_if(self, interface):

        if_split = tf.split(interface, self.if_sizes, axis=1)

        # Read key and read mode start and end indices.
        rk_start, rk_end = 6, 6 + self.read_heads
        rm_start, rm_end = rk_end, rk_end + self.read_heads

        return {
            'write_key': if_split[0],
            'write_vec': if_split[1],
            'erase_vec': tf.math.sigmoid(if_split[2]),
            'free_gate': tf.math.sigmoid(if_split[3]),
            'read_strength': self.__oneplus(if_split[4]),
            'write_strength': self.__oneplus(if_split[5][:, 0]),
            'write_gate': tf.math.sigmoid(if_split[5][:, 1]),
            'alloc_gate': tf.math.sigmoid(if_split[5][:, 2]),
            'read_keys': [if_split[i] for i in range(rk_start, rk_end)],
            'read_mode': [if_split[i] for i in range(rm_start, rm_end)]
        }


class TemporalLink(Model):

    def __init__(self, batch_size, memory_size):

        super(TemporalLink, self).__init__()
        self.matrix = tf.zeros((
            batch_size,
            memory_size,
            memory_size
        ))

    def update(self, precedence, write_w):

        shape = (*write_w.shape, write_w.shape[1])
        write_w = tf.expand_dims(write_w, axis=-1)
        precedence = tf.expand_dims(precedence, axis=1)

        bc_write_w = tf.broadcast_to(write_w, shape)
        bc_write_w_t = tf.transpose(bc_write_w, perm=[0, 2, 1])
        temp1 = tf.subtract(bc_write_w, bc_write_w_t)
        temp1 = tf.subtract(tf.ones(shape), temp1)
        temp1 = tf.multiply(temp1, self.matrix)

        bc_precedence = tf.broadcast_to(precedence, shape)
        temp2 = tf.multiply(bc_write_w, bc_precedence)
        updated = tf.add(temp1, temp2)

        # Ensure self-links are excluded.
        updated = tf.linalg.set_diag(updated)
        self.matrix = updated

    def __precedence_weight(self, prev_precedence, write_w):

        ones = tf.ones((write_w.shape[0], 1))
        sum_write_w = tf.reduce_sum(write_w, axis=1, keep_dims=True)
        temp = tf.subtract(ones, sum_write_w)
        temp = tf.multiply(temp, prev_precedence)
        precedence = tf.add(temp, write_w)
        return precedence


class FFN(Model):

    def __init__(self, opts):

        super(FFN, self).__init__()
        self.__layers = []

        # Hidden layers.
        for hid_size in opts.hid_sizes:
            dense = Dense(hid_size, 'relu')
            self.__layers.append(dense)

        # Output layer.
        self.__out_layer = Dense(
            opts.output_size,
            activation=None,
            use_bias=False
        )

        # Interface layer.
        self.__interface = Dense(
            opts.interface_size,
            activation=None,
            use_bias=False
        )

    def __call__(self, inputs, state):

        hidden = inputs
        for layer in self.__layers:
            hidden = layer(hidden)

        ctrl_output = {
            'output': self.__out_layer(hidden),
            'interface': self.__interface(hidden)
        }

        return (ctrl_output, state)

    def initialize(self, batch_size):

        # No state.
        return None


class LSTM(Model):

    def __init__(self, opts):

        super(LSTM, self).__init__()
        self.cell = LSTMCell(opts.lstm_units)

        # Output layer.
        self.__out_layer = Dense(
            opts.output_size,
            activation=None,
            use_bias=False
        )

        # Interface layer.
        self.__interface = Dense(
            opts.interface_size,
            activation=None,
            use_bias=False
        )

    def __call__(self, inputs, state):

        hidden, state = self.cell(inputs, state)

        ctrl_output = {
            'output': self.__out_layer(hidden),
            'interface': self.__interface(hidden)
        }

        return ctrl_output, state

    def initialize(self, batch_size):

        return self.cell.get_initial_state(
            batch_size=batch_size,
            dtype=tf.float32
        )
