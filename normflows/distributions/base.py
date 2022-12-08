from matplotlib.pyplot import sca
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
from torch.distributions import ExponentialFamily,Categorical,constraints,MultivariateNormal, Independent
from torch.distributions.utils import _standard_normal,broadcast_all
from numbers import Real, Number
import math
import copy
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
import torch
from pyro.distributions import Chi2, TorchDistribution
from torch._six import inf, nan
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, broadcast_all, lazy_property
from typing import Dict
from scipy.stats import gennorm
import torch
from torch.distributions import Gamma, MultivariateNormal, StudentT
from numbers import Number
import torch
from torch.distributions import constraints, Gamma, MultivariateNormal
from torch.distributions.multivariate_normal import _batch_mv, _batch_mahalanobis
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, _standard_normal
from scipy import stats
import math


from pyro.distributions import MultivariateStudentT
from .. import flows


class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """
        Samples from base distribution and calculates log probability
        :param num_samples: Number of samples to draw from the distriubtion
        :return: Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """
        Calculate log probability of batch of samples
        :param z: Batch of random variables to determine log probability for
        :return: log probability for each batch element
        """
        raise NotImplementedError


class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, trainable=True):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p


class UniformGaussian(BaseDistribution):
    """
    Distribution of a 1D random variable with some entries having a uniform and
    others a Gaussian distribution
    """

    def __init__(self, ndim, ind, scale=None):
        """
        Constructor
        :param ndim: Int, number of dimensions
        :param ind: Iterable, indices of uniformly distributed entries
        :param scale: Iterable, standard deviation of Gaussian or width of
        uniform distribution
        """
        super().__init__()
        self.ndim = ndim

        # Set up indices and permutations
        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer("ind", torch._cast_Long(ind))
        else:
            self.register_buffer("ind", torch.tensor(ind, dtype=torch.long))

        ind_ = []
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer("ind_", torch.tensor(ind_, dtype=torch.long))

        perm_ = torch.cat((self.ind, self.ind_))
        inv_perm_ = torch.zeros_like(perm_)
        for i in range(self.ndim):
            inv_perm_[perm_[i]] = i
        self.register_buffer("inv_perm", inv_perm_)

        if scale is None:
            self.register_buffer("scale", torch.ones(self.ndim))
        else:
            self.register_buffer("scale", scale)

    def forward(self, num_samples=1):
        z = self.sample(num_samples)
        return z, self.log_prob(z)

    def sample(self, num_samples=1):
        eps_u = (
            torch.rand(
                (num_samples, len(self.ind)),
                dtype=self.scale.dtype,
                device=self.scale.device,
            )
            - 0.5
        )
        eps_g = torch.randn(
            (num_samples, len(self.ind_)),
            dtype=self.scale.dtype,
            device=self.scale.device,
        )
        z = torch.cat((eps_u, eps_g), -1)
        z = z[..., self.inv_perm]
        return self.scale * z

    def log_prob(self, z):
        log_p_u = torch.broadcast_to(-torch.log(self.scale[self.ind]), (len(z), -1))
        log_p_g = (
            -0.5 * np.log(2 * np.pi)
            - torch.log(self.scale[self.ind_])
            - 0.5 * torch.pow(z[..., self.ind_] / self.scale[self.ind_], 2)
        )
        return torch.sum(log_p_u, -1) + torch.sum(log_p_g, -1)


class ClassCondDiagGaussian(BaseDistribution):
    """
    Class conditional multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, num_classes):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        :param num_classes: Number of classes to condition on
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.perm = [self.n_dim] + list(range(self.n_dim))
        self.d = np.prod(shape)
        self.num_classes = num_classes
        self.loc = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.log_scale = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, y=None):
        if y is not None:
            num_samples = len(y)
        else:
            y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
        if y.dim() == 1:
            y_onehot = torch.zeros(
                (self.num_classes, num_samples),
                dtype=self.loc.dtype,
                device=self.loc.device,
            )
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        z = loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, y):
        if y.dim() == 1:
            y_onehot = torch.zeros(
                (self.num_classes, len(y)), dtype=self.loc.dtype, device=self.loc.device
            )
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p


class GlowBase(BaseDistribution):
    """
    Base distribution of the Glow model, i.e. Diagonal Gaussian with one mean and
    log scale for each channel
    """

    def __init__(self, shape, num_classes=None, logscale_factor=3.0):
        """
        Constructor
        :param shape: Shape of the variables
        :param num_classes: Number of classes if the base is class conditional,
        None otherwise
        :param logscale_factor: Scaling factor for mean and log variance
        """
        super().__init__()
        # Save shape and related statistics
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.num_pix = np.prod(shape[1:])
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        self.logscale_factor = logscale_factor
        # Set up parameters
        self.loc = nn.Parameter(
            torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1]))
        )
        self.loc_logs = nn.Parameter(
            torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1]))
        )
        self.log_scale = nn.Parameter(
            torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1]))
        )
        self.log_scale_logs = nn.Parameter(
            torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1]))
        )
        # Class conditional parameter if needed
        if self.class_cond:
            self.loc_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
            self.log_scale_cc = nn.Parameter(
                torch.zeros(self.num_classes, self.shape[0])
            )
        # Temperature parameter for annealed sampling
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        # Prepare parameter
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(
            self.log_scale_logs * self.logscale_factor
        )
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(
                    self.num_classes, (num_samples,), device=self.loc.device
                )
            if y.dim() == 1:
                y_onehot = torch.zeros(
                    (len(y), self.num_classes),
                    dtype=self.loc.dtype,
                    device=self.loc.device,
                )
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(
                y.size(0), self.shape[0], *((self.n_dim - 1) * [1])
            )
            log_scale = log_scale + (y @ self.log_scale_cc).view(
                y.size(0), self.shape[0], *((self.n_dim - 1) * [1])
            )
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        # Sample
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        z = loc + torch.exp(log_scale) * eps
        # Get log prob
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - self.num_pix * torch.sum(log_scale, dim=self.sum_dim)
            - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        )
        return z, log_p

    def log_prob(self, z, y=None):
        # Perpare parameter
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(
            self.log_scale_logs * self.logscale_factor
        )
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros(
                    (len(y), self.num_classes),
                    dtype=self.loc.dtype,
                    device=self.loc.device,
                )
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(
                y.size(0), self.shape[0], *((self.n_dim - 1) * [1])
            )
            log_scale = log_scale + (y @ self.log_scale_cc).view(
                y.size(0), self.shape[0], *((self.n_dim - 1) * [1])
            )
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        # Get log prob
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - self.num_pix * torch.sum(log_scale, dim=self.sum_dim)
            - 0.5
            * torch.sum(
                torch.pow((z - loc) / torch.exp(log_scale), 2), dim=self.sum_dim
            )
        )
        return log_p


class AffineGaussian(BaseDistribution):
    """
    Diagonal Gaussian an affine constant transformation applied to it,
    can be class conditional or not
    """

    def __init__(self, shape, affine_shape, num_classes=None):
        """
        Constructor
        :param shape: Shape of the variables
        :param affine_shape: Shape of the parameters in the affine transformation
        :param num_classes: Number of classes if the base is class conditional,
        None otherwise
        """
        super().__init__()
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.affine_shape = affine_shape
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        # Affine transformation
        if self.class_cond:
            self.transform = flows.CCAffineConst(self.affine_shape, self.num_classes)
        else:
            self.transform = flows.AffineConstFlow(self.affine_shape)
        # Temperature parameter for annealed sampling
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        dtype = self.transform.s.dtype
        device = self.transform.s.device
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=device)
            if y.dim() == 1:
                y_onehot = torch.zeros(
                    (len(y), self.num_classes), dtype=dtype, device=device
                )
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.0
        # Sample
        eps = torch.randn((num_samples,) + self.shape, dtype=dtype, device=device)
        z = np.exp(log_scale) * eps
        # Get log prob
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            - self.d * log_scale
            - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        )
        # Apply transform
        if self.class_cond:
            z, log_det = self.transform(z, y)
        else:
            z, log_det = self.transform(z)
        log_p -= log_det
        return z, log_p

    def log_prob(self, z, y=None):
        # Perpare onehot encoding of class if needed
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros(
                    (len(y), self.num_classes),
                    dtype=self.transform.s.dtype,
                    device=self.transform.s.device,
                )
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.0
        # Get log prob
        if self.class_cond:
            z, log_p = self.transform.inverse(z, y)
        else:
            z, log_p = self.transform.inverse(z)
        z = z / np.exp(log_scale)
        log_p = (
            log_p
            - self.d * log_scale
            - 0.5 * self.d * np.log(2 * np.pi)
            - 0.5 * torch.sum(torch.pow(z, 2), dim=self.sum_dim)
        )
        return log_p



class GaussianPCA(BaseDistribution):
    """
    Gaussian distribution resulting from linearly mapping a normal distributed latent
    variable describing the "content of the target"
    """

    def __init__(self, dim, latent_dim=None, sigma=0.1):
        """
        Constructor
        :param dim: Number of dimensions of the flow variables
        :param latent_dim: Number of dimensions of the latent "content" variable;
                           if None it is set equal to dim
        :param sigma: Noise level
        """
        super().__init__()

        self.dim = dim
        if latent_dim is None:
            self.latent_dim = dim
        else:
            self.latent_dim = latent_dim

        self.loc = nn.Parameter(torch.ones(1, dim))
        self.W = nn.Parameter(torch.randn(latent_dim, dim))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))

    def forward(self, num_samples=1):
        eps = torch.randn(
            num_samples, self.latent_dim, dtype=self.loc.dtype, device=self.loc.device
        )
        z_ = torch.matmul(eps, self.W)
        z = z_ + self.loc

        Sig = torch.matmul(self.W.T, self.W) + torch.exp(
            self.log_sigma * 2
        ) * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = (
            self.dim / 2 * np.log(2 * np.pi)
            - 0.5 * torch.det(Sig)
            - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        )

        return z, log_p

    def log_prob(self, z):
        z_ = z - self.loc

        Sig = torch.matmul(self.W.T, self.W) + torch.exp(
            self.log_sigma * 2
        ) * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = (
            self.dim / 2 * np.log(2 * np.pi)
            - 0.5 * torch.det(Sig)
            - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        )

        return log_p




class GeneralizedNormal(Distribution):
    r"""
    Creates a Generalized Normal distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`beta`.
    Example::
        >>> m = GeneralizedNormal(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor(0.5))
        >>> m.sample()  # GeneralizedNormal distributed with loc=0, scale=1, beta=0.5
        tensor([ 0.1337])
    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
        beta (float or Tensor): shape parameter of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'beta': constraints.positive}
    support = constraints.real
    has_rsample = False

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale.pow(2) * (torch.lgamma(3/self.beta) - torch.lgamma(1/self.beta)).exp()

    @property
    def stddev(self):
        return self.variance()**0.5

    def __init__(self, loc=5., scale=5., beta=2., validate_args=None):

        self.loc, self.scale = broadcast_all(loc, scale)
        (self.beta,) = broadcast_all(beta)
        self.scipy_dist = stats.gennorm(loc=self.loc.cpu().detach().numpy(),
                            scale=self.scale.cpu().detach().numpy(),
                            beta=self.beta.cpu().detach().numpy())
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(GeneralizedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(GeneralizedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new




    def sample(self, sample_shape):
        sample_shape = torch.Size(sample_shape)
        sample_shape = sample_shape + self.loc.size()
        return torch.tensor(self.scipy_dist.rvs(
            list(sample_shape),
            random_state=torch.randint(2**32, ()).item()),  # Make deterministic if torch is seeded
                            dtype=self.loc.dtype, device=self.loc.device)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (-torch.log(2 * self.scale) - torch.lgamma(1/self.beta) + torch.log(self.beta)
                - torch.pow((torch.abs(value - self.loc) / self.scale), self.beta))

    # def forward(self, num_samples=1):
    #     #print('~~~1',self.gmm.mixture_distribution.probs)
        
    #     z = self.sample([num_samples])
    #     log_prob= self.log_prob(z)
    #     return z, log_prob

    def cdf(self, value):
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        return torch.tensor(self.scipy_dist.cdf(value),
                            dtype=self.loc.dtype, device=self.loc.device)


    def icdf(self, value):
        raise NotImplementedError


    def entropy(self):
        return (1/self.beta) - torch.log(self.beta) + torch.log(2*self.scale) + torch.lgamma(1/self.beta)




class MixtureofGaussians(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True, loc_mult = 5.0):
        """
        Constructor
        :param n_modes: Number of modes of the mixture model
        :param dim: Number of dimensions of each Gaussian
        :param loc: List of mean values
        :param scale: List of diagonals of the covariance matrices
        :param weights: List of mode probabilities
        :param trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        # if loc is None:
        #     loc = np.random.randn(self.n_modes, self.dim) * loc_mult
        # loc = np.array(loc)[None, ...]
        # if scale is None:
        #     scale = np.random.rand(self.n_modes, self.dim) * loc_mult
        if loc is None:
            loc = np.random.rand(self.n_modes, self.dim) * loc_mult
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.random.rand(self.n_modes, self.dim) * loc_mult
        #print(loc,scale)
            

        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.random.rand(self.n_modes)+ 1.
            
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))

    def forward(self, num_samples=1):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Sample mode indices
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]

        # Get samples
        eps_ = torch.randn(
            num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device
        )
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return log_p



class T(BaseDistribution):
    """
    Normal T
    """
    def __init__(self, n_dim=2, df = 2., loc = 5., scale = 2., trainable=False):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        
        self.n_dim = n_dim
        


        with torch.no_grad():
            if trainable:
                self.df = nn.Parameter(torch.rand(self.n_dim,dtype=torch.double,device='cuda') + df, requires_grad = True)
                self.loc = nn.Parameter(torch.randn(self.n_dim,dtype=torch.double,device='cuda') * loc, requires_grad = True)
                self.scale = nn.Parameter(torch.rand(self.n_dim,dtype=torch.double,device='cuda') * scale, requires_grad = True)

            else:
                self.register_buffer("df", torch.ones((self.n_dim,),dtype=torch.double,device='cuda') + df)

        self.t = D.StudentT(df=self.df, loc=self.loc, scale=self.scale)#univ

        
        #print('=========sample',self.t.log_prob(self.t.sample([1000])).mean(1).shape)


    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """

        return self.t.sample([num_samples])

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)
        return self.t.log_prob(z).mean(1)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.sample(num_samples)
        log_prob= self.log_prob(z)
        return z, log_prob

class TMV(BaseDistribution):
    """
    Multivariate T
    """
    def __init__(self, n_dim=2, df = 3., trainable=False):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        
        self.n_dim = n_dim
        rank = n_dim + n_dim
        tdf = torch.rand((1,), device='cuda').exp()
        tdf = tdf - tdf + df
        
        tloc = torch.randn((1,) + (n_dim,), device='cuda')
        tcov = torch.randn((1,) + (n_dim, rank),device='cuda')
        tcov = tcov.matmul(tcov.transpose(-1, -2))
        tscale_tril = tcov.cholesky()        


        with torch.no_grad():
            if trainable:
                self.df = nn.Parameter(tdf.double(), requires_grad = True)
                self.loc = nn.Parameter(tloc.double(), requires_grad = True)
                self.scale = nn.Parameter(tscale_tril.double(), requires_grad = True)

                
            else:
                self.register_buffer("df", tdf)
                self.register_buffer("loc", tloc)
                self.register_buffer("scale", tscale_tril)

        self.t = MultivariateStudentT(self.df, self.loc, self.scale)#univ
        #print('~~~1',self.gmm.mixture_distribution.probs)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        #print(self.t.loc.dtype,self.t.df.dtype,self.t.scale_tril.dtype)
        
        z = self.t.sample([num_samples])
        
        
        log_prob= self.t.log_prob(z).mean(axis=1)
        #print(z.shape,log_prob.shape)
        return z.squeeze(), log_prob

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)

        return self.t.log_prob(z[:,None,:]).mean(axis=1)


class MultivariateGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, trainable=False, loc = 5., scale = 2.):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        
        self.n_dim = n_dim

        with torch.no_grad():
            if trainable:
                
                self.loc = nn.Parameter(torch.randn(self.n_dim).double().cuda() * loc)
                Sigma_k = torch.rand(self.n_dim, self.n_dim) * scale
                Sigma_k = torch.mm(Sigma_k, Sigma_k.t()).cuda()
                #Sigma_k.add_(torch.eye(512))
                self.scale = nn.Parameter(torch.eye(self.n_dim,dtype=torch.double,device='cuda') + Sigma_k, requires_grad = True)
            else:
                
                self.register_buffer("loc", torch.zeros(self.n_dim,dtype=torch.double,device='cuda')+loc)
                self.register_buffer("scale", torch.eye(self.n_dim,dtype=torch.double,device='cuda')+scale)

        self.mvn = D.MultivariateNormal(self.loc, self.scale)
        
    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.mvn.sample([num_samples])
        #print(z)
        log_prob= self.mvn.log_prob(z)
        return z, log_prob

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)

        return self.mvn.log_prob(z)






class MixtureofGeneralizedGaussians(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, n_components = 3, beta=2.,loc=5.,scale=2., trainable=False, prop_scale=torch.tensor(6.),
                 prop_shift=torch.tensor(-3.)):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        self.n_components = n_components
        self.n_dim = n_dim
        self.beta = beta
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)
        self.max_log_prob = 0.

        self.gmm = []
        with torch.no_grad():
            for _ in range(self.n_dim):
                if trainable:

                    self.w = nn.Parameter(torch.rand(self.n_components).cuda())
                    self.loc = nn.Parameter(torch.randn([self.n_components,self.n_dim]).cuda() * loc)
                    self.scale = nn.Parameter(torch.rand([self.n_components,self.n_dim]).cuda() * scale)
                    self.p = nn.Parameter(torch.rand(self.n_dim).cuda()+self.beta,requires_grad=True)
                    self.mix = D.Categorical(self.w)
                    self.comp = D.Independent(GeneralizedNormal(self.loc, self.scale,self.p), 1)
                    self.gmm = D.MixtureSameFamily(self.mix, self.comp)
                    #print('shape===============',self.gmm.sample([1000]).shape,self.gmm.log_prob(self.gmm.sample([1000])).shape)

                # else:
                #     self.register_buffer("w", torch.rand(self.n_components))
                #     self.mix = D.Categorical(self.w)
                #     self.register_buffer("loc", torch.rand([self.n_components,self.n_dim]))
                #     self.register_buffer("scale", torch.rand([self.n_components,self.n_dim]))
                #     self.register_buffer("p", torch.tensor(self.beta))
                #     self.comp = D.Independent(GenNormal(self.loc, self.scale,self.p), 1)
                #     self.gmm = D.MixtureSameFamily(self.mix, self.comp)
                    



    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """

        return self.gmm.sample([num_samples])

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)
        return self.gmm.log_prob(z)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.sample(num_samples)
        log_prob= self.log_prob(z)
        return z, log_prob

class GeneralizedGaussiansDistribution(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, beta=2.,loc=5.,scale=2., trainable=False, prop_scale=torch.tensor(6.),
                 prop_shift=torch.tensor(-3.)):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        
        self.n_dim = n_dim
        self.beta = beta
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)
        self.max_log_prob = 0.

        self.gmm = []
        with torch.no_grad():
            for _ in range(self.n_dim):
                if trainable:

                    self.loc = nn.Parameter(torch.randn(self.n_dim).cuda() * loc)
                    self.scale = nn.Parameter(torch.rand(self.n_dim).cuda() * scale)
                    self.beta = nn.Parameter(torch.rand(self.n_dim).cuda()+self.beta,requires_grad=True)
                    self.gmm = GeneralizedNormal(loc=self.loc, scale=self.scale,beta=self.beta)
                    #print('shape===============',self.gmm.sample([1000]).shape,self.gmm.log_prob(self.gmm.sample([1000])).shape)

                # else:
                #     self.register_buffer("w", torch.rand(self.n_components))
                #     self.mix = D.Categorical(self.w)
                #     self.register_buffer("loc", torch.rand([self.n_components,self.n_dim]))
                #     self.register_buffer("scale", torch.rand([self.n_components,self.n_dim]))
                #     self.register_buffer("p", torch.tensor(self.beta))
                #     self.comp = D.Independent(GenNormal(self.loc, self.scale,self.p), 1)
                #     self.gmm = D.MixtureSameFamily(self.mix, self.comp)
                    



    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """

        return self.gmm.sample([num_samples])

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)
        return self.gmm.log_prob(z).mean(1)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.sample(num_samples)
        log_prob= self.log_prob(z)
        return z, log_prob


class MixtureofT(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, n_dim=2, n_components = 3, df=2.,loc=5.,scale=2., trainable=False, prop_scale=torch.tensor(6.),
                 prop_shift=torch.tensor(-3.)):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()

        self.n_components = n_components
        self.n_dim = n_dim
        self.df = df
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)
        self.max_log_prob = 0.

        self.gmm = []
        with torch.no_grad():
            for _ in range(self.n_dim):
                if trainable:

                    self.w = nn.Parameter(torch.rand(self.n_components).cuda())
                    self.loc = nn.Parameter(torch.randn([self.n_components,self.n_dim]).cuda() * loc)
                    self.scale = nn.Parameter(torch.rand([self.n_components,self.n_dim]).cuda() * scale)
                    self.df = nn.Parameter(torch.rand(self.n_dim).cuda()+self.df,requires_grad=True)
                    self.mix = D.Categorical(self.w)
                    #univ
                    self.comp = D.Independent(D.StudentT(df=self.df, loc=self.loc, scale=self.scale), 1)
                    self.gmm = D.MixtureSameFamily(self.mix, self.comp)
                    #print('shape===============',self.gmm.sample([1000]).shape,self.gmm.log_prob(self.gmm.sample([1000])).shape)

                # else:
                #     self.register_buffer("w", torch.rand(self.n_components))
                #     self.mix = D.Categorical(self.w)
                #     self.register_buffer("loc", torch.rand([self.n_components,self.n_dim]))
                #     self.register_buffer("scale", torch.rand([self.n_components,self.n_dim]))
                #     self.register_buffer("p", torch.tensor(self.beta))
                #     self.comp = D.Independent(GenNormal(self.loc, self.scale,self.p), 1)
                #     self.gmm = D.MixtureSameFamily(self.mix, self.comp)
                    



    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """

        return self.gmm.sample([num_samples])

    def log_prob(self, z):
        #print('~~~0',self.loc.is_leaf,self.scale.is_leaf,self.w.is_leaf)
        return self.gmm.log_prob(z)


    def forward(self, num_samples=1):
        #print('~~~1',self.gmm.mixture_distribution.probs)
        
        z = self.sample(num_samples)
        log_prob= self.log_prob(z)
        return z, log_prob


