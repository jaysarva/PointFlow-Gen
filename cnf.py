import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

__all__ = ["CNF", "SequentialFlow"]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx, reverse, inds, integration_times):
        for i in inds:
            x, logpx = self.chain[i](x, context, logpx, integration_times, reverse)
        return x, logpx


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False):
        super(CNF, self).__init__()
        self.T = 1.0
        self.use_adjoint = True
        self.odefunc = odefunc
        self.solver = 'dopri5'
        self.atol = 1e-5
        self.rtol = 1e-5
        self.test_solver = 'dopri5'
        self.conditional = True

    def forward(self, x, context, integration_times):
        logpx = torch.zeros(*x.shape[:-1], 1).to(x)

        states = (x, logpx, context)
        atol = [self.atol] * 3
        rtol = [self.rtol] * 3
        
        integration_times = torch.stack([torch.tensor(0.0).to(x), self.sqrt_end_time * self.sqrt_end_time]).to(x)

        self.odefunc.before_odeint()
        odeint = odeint_adjoint
        state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver)
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]

        return z_t, logpz_t
        
    def num_evals(self):
        return self.odefunc._num_evals.item()
