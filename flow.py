from ode import ODEfunc, ODEnet
from normalize import MovingBatchNorm1d
from cnf import CNF, SequentialFlow


def get_point_cnf(args):
    dims = tuple(map(int, args.dims.split("-")))
    model = build_model(args, args.input_dim, dims, args.zdim, args.num_blocks, True)
    return model


def get_latent_cnf(args):
    dims = tuple(map(int, args.latent_dims.split("-")))
    model = build_model(args, args.zdim, dims, 0, args.latent_num_blocks)
    return model

def build_model(input_dim, hidden_dims, context_dim, num_blocks):
    def build_cnf():
        diffeq = ODEnet(hidden_dims=hidden_dims, input_shape=(input_dim,), context_dim=context_dim)
        odefunc = ODEfunc(diffeq=diffeq)
        cnf = CNF(odefunc=odefunc)
        return cnf

    chain = [build_cnf() for i in range(num_blocks)]
    bn_layers = [MovingBatchNorm1d(input_dim) for _ in range(num_blocks)]
    bn_chain = [MovingBatchNorm1d(input_dim)]
    for a, b in zip(chain, bn_layers):
        bn_chain.append(a)
        bn_chain.append(b)
    chain = bn_chain
    model = SequentialFlow(chain)

    return model.cuda()


def count_nfe(model):
    class AccEvals(object):
        def __init__(self):
            self.evals = 0

        def __call__(self, m):
            if isinstance(m, CNF):
                self.evals += m.evals()

    acc = AccEvals()
    model.apply(acc)
    return acc.evals


def count_parameters(m):
    return sum(param.numel() for param in m.parameters() if param.requires_grad)


def count_total_time(model):
    class Accumulator(object):
        def __init__(self):
            self.time = 0

        def __call__(self, m):
            if isinstance(m, CNF):
                self.time += + m.sqrt_end_time * m.sqrt_end_time

    acc = Accumulator()
    model.apply(acc)
    return acc.total_time
