import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal
from scipy.stats import multivariate_t, lognorm, cauchy, invweibull, pareto, chi2
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import torch.distributions as D

class Target(nn.Module):
    """
    Sample target distributions to test models
    """

    def __init__(self, prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)):
        """
        Constructor
        :param prop_scale: Scale for the uniform proposal
        :param prop_shift: Shift for the uniform proposal
        """
        super().__init__()
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        raise NotImplementedError("The log probability is not implemented yet.")

    def rejection_sampling(self, num_steps=1):
        """
        Perform rejection sampling on image distribution
        :param num_steps: Number of rejection sampling steps to perform
        :return: Accepted samples
        """
        eps = torch.rand(
            (num_steps, self.n_dims),
            dtype=self.prop_scale.dtype,
            device=self.prop_scale.device,
        )
        z_ = self.prop_scale * eps + self.prop_shift
        prob = torch.rand(
            num_steps, dtype=self.prop_scale.dtype, device=self.prop_scale.device
        )
        prob_ = torch.exp(self.log_prob(z_) - self.max_log_prob)
        accept = prob_ > prob
        z = z_[accept, :]
        return z

    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        z = torch.zeros(
            (0, self.n_dims), dtype=self.prop_scale.dtype, device=self.prop_scale.device
        )
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z

class NealsFunnel(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,prop_scale=torch.tensor(20.),
                 prop_shift=torch.tensor(-10.), v1shift = 0., v2shift = 0.):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.
        self.v1shift = v1shift
        self.v2shift = v2shift
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)


    # def sample(self, num_samples=1):
    #     """
    #     :param num_samples: Number of samples to draw
    #     :return: Samples
    #     """
    #     data = []
    #     n_dims = 1
    #     for i in range(nsamples):
    #         v = norm(0, 1).rvs(1)
    #         x = norm(0, np.exp(0.5*v)).rvs(n_dims)
    #         data.append(np.hstack([v, x]))
    #     data = pd.DataFrame(data)
    #     return torch.tensor(data.values)

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        #print('++++++++++',z)
        v = z[:,0].cpu()
        x = z[:,1].cpu()
        v_like = Normal(torch.tensor([0.0]).cpu(), torch.tensor([1.0]).cpu() + self.v1shift).log_prob(v).cpu()
        x_like = Normal(torch.tensor([0.0]).cpu(), torch.exp(0.5*v).cpu() + self.v2shift).log_prob(x).cpu()
        return v_like + x_like


class StudentTDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,df=2.,dim=2):
        super().__init__()
        self.df = df
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(multivariate_t(loc=self.loc,df=self.df).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(multivariate_t(loc=self.loc,df=self.df).logpdf(z.cpu().detach().numpy()),device='cuda')



class chi2Dist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,df=2.,dim=2):
        super().__init__()
        self.df = df
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(chi2(loc=self.loc,df=self.df).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(chi2(loc=self.loc,df=self.df).logpdf(z.cpu().detach().numpy()),device='cuda')


class FrechetDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,c=2.,dim=2):
        super().__init__()
        self.c = c
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(invweibull(loc=self.loc,c=self.c).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(invweibull(loc=self.loc,c=self.c).logpdf(z.cpu().detach().numpy()),device='cuda')


class ParetoDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,b=2.,dim=2):
        super().__init__()
        self.b = b
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(pareto(loc=self.loc,b=self.b).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(pareto(loc=self.loc,b=self.b).logpdf(z.cpu().detach().numpy()),device='cuda')


class CauchyDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,dim=2):
        super().__init__()
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(cauchy(loc=self.loc).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(cauchy(loc=self.loc).logpdf(z.cpu().detach().numpy()),device='cuda')


class LogNormDist(Target):
    """
    Bimodal two-dimensional distribution
    """
    def __init__(self,s=2.,dim=2):
        super().__init__()
        self.s = s
        self.loc = np.repeat([0.],dim)

    def sample(self, num_samples=1):
        """
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return torch.tensor(lognorm(loc=self.loc,s=self.s).rvs(num_samples),device='cuda')

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        return torch.tensor(lognorm(loc=self.loc,s=self.s).logpdf(z.cpu().detach().numpy()),device='cuda')


class TwoMoons(Target):
    """
    Bimodal two-dimensional distribution
    """

    def __init__(self):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0

    def log_prob(self, z):
        """
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )
        return log_prob


class CircularGaussianMixture(nn.Module):
    """
    Two-dimensional Gaussian mixture arranged in a circle
    """

    def __init__(self, n_modes=8):
        """
        Constructor
        :param n_modes: Number of modes
        """
        super(CircularGaussianMixture, self).__init__()
        self.n_modes = n_modes
        self.register_buffer(
            "scale", torch.tensor(2 / 3 * np.sin(np.pi / self.n_modes)).float()
        )

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_modes):
            d_ = (
                (z[:, 0] - 2 * np.sin(2 * np.pi / self.n_modes * i)) ** 2
                + (z[:, 1] - 2 * np.cos(2 * np.pi / self.n_modes * i)) ** 2
            ) / (2 * self.scale**2)
            d = torch.cat((d, d_[:, None]), 1)
        log_p = -torch.log(
            2 * np.pi * self.scale**2 * self.n_modes
        ) + torch.logsumexp(-d, 1)
        return log_p

    def sample(self, num_samples=1):
        eps = torch.randn(
            (num_samples, 2), dtype=self.scale.dtype, device=self.scale.device
        )
        phi = (
            2
            * np.pi
            / self.n_modes
            * torch.randint(0, self.n_modes, (num_samples,), device=self.scale.device)
        )
        loc = torch.stack((2 * torch.sin(phi), 2 * torch.cos(phi)), 1).type(eps.dtype)
        return eps * self.scale + loc


class RingMixture(Target):
    """
    Mixture of ring distributions in two dimensions
    """

    def __init__(self, n_rings=2):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0
        self.n_rings = n_rings
        self.scale = 1 / 4 / self.n_rings

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_rings):
            d_ = ((torch.norm(z, dim=1) - 2 / self.n_rings * (i + 1)) ** 2) / (
                2 * self.scale**2
            )
            d = torch.cat((d, d_[:, None]), 1)
        return torch.logsumexp(-d, 1)
