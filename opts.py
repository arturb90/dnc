def get_opts(args):

    return {
        'session': SessionOpts(args),
        'model': ModelOpts(args),
        'task': TaskOpts(args)
    }


class SessionOpts:

    def __init__(self, args):

        self.batch_size = args.batch_size
        self.epochs = args.epochs


class ModelOpts:

    input_size = {
        'copyrepeat': (lambda args: args.vec_len)
    }

    output_size = {
        'copyrepeat': (lambda args: args.vec_len)
    }

    def __init__(self, args):

        self.controller = args.controller
        self.memory_size = args.memory_size
        self.address_size = args.address_size
        self.read_heads = args.read_heads
        self.lstm_units = args.lstm_units
        self.hid_sizes = args.hid_sizes

        self.input_size = ModelOpts.input_size[args.task](args)
        self.output_size = ModelOpts.output_size[args.task](args)
        self.interface_size = args.read_heads * args.address_size
        self.interface_size += (3 * args.address_size) + 3
        self.interface_size += 5 * args.read_heads


class TaskOpts:

    def __init__(self, args):

        self.task = args.task
        self.seq_len = args.seq_len
        self.vec_len = args.vec_len
        self.n_pairs = args.n_pairs
