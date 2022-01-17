import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from flow import get_point_cnf
from flow import get_latent_cnf
import math

def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def reduce_tensor(tensor, world_size=None):
    temp = tensor.clone()
    torch.distributed.all_reduce(temp, op=torch.distributed.ReduceOp.SUM)
    if world_size is None:
        world_size = torch.distributed.get_world_size()

    temp /= world_size
    return temp


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * math.log(2 * math.pi)
    return log_z - z.pow(2) / 2

class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        return m, v


# Model
class PointFlow(nn.Module):
    def __init__(self, args):
        super(PointFlow, self).__init__()
        self.input_dim = 512
        self.zdim = 128
        self.prior_weight = 0.4
        self.recon_weight = 0.6
        self.entropy_weight = 0.6
        self.distributed = True
        self.truncate_std = None
        self.encoder = Encoder(zdim= 128, input_dim = 512)
        self.point_cnf = get_point_cnf(args)
        self.latent_cnf = get_latent_cnf(args)

        self.optimizer = torch.optim.Adam(learning_rate=2e-3, beta=(0.9,0.999))
        self.batch_size = 16


    def sample_gaussian(self, size, gpu):
        y = torch.randn(*size).float()
        y = y.cuda(gpu)
        truncated_normal(y, mean=0, std=1, trunc_std=self.truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def forward(self, x):
        self.opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        z = self.reparameterize_gaussian(z_mu, z_sigma)

        # getting entropy val
        entropy = self.gaussian_entropy(z_sigma)

        # prior probability P(z)
        w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_pw = delta_log_pw.view(batch_size, 1)
        log_pz = log_pw - delta_log_pw

        # reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss
        loss.backward()
        self.opt.step()

        entropy_log = entropy.mean()
        
        return entropy_log

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
        w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
        z = self.latent_cnf(w, None, reverse=True).view(*w.size())
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, num_points):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, self.truncate_std)
        return x