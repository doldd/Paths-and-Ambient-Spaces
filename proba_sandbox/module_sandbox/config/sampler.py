"""Sampler Configuration."""
import warnings
from dataclasses import dataclass, field
from typing import Any

from module_sandbox.config.base import BaseConfig, BaseStrEnum
from module_sandbox.training.priors import PriorDist


class Sampler(BaseStrEnum):
    """Sampler Names.

    Notes
    -----
    - This enum class defines the samplers that can be used for Bayesian Inference.
    - To extend the possible samplers, add a new value to the `Sampler` enum.
    - The `get_kernel` and `get_warmup_kernel` methods are used to get the kernel
        and warmup kernel for the sampler respectively.

    - NOTE: The `get_kernel` and `get_warmup_kernel` methods return the kernel
        not the sampler itself. The sampler is later initialized in the
        `module_sandbox.training.trainer` module.
    """

    NUTS = 'nuts'
    MCLMC = 'mclmc'
    HMC = 'hmc'
    SGLD = 'sgld'
    ADASGHMC = 'adasghmc'
    SGHMC = 'sghmc'

    def get_kernel(self):
        """Get sampling kernel."""
        from module_sandbox.training.kernels import KERNELS

        if self.value not in KERNELS:
            raise NotImplementedError(
                f'Sampler for {self.value} is not yet implemented.'
            )
        return KERNELS[self.value]

    def get_warmup_kernel(self):
        """Get warmup kernel."""
        from module_sandbox.training.kernels import WARMUP_KERNELS

        if self.value not in WARMUP_KERNELS:
            raise NotImplementedError(
                f'Warmup Kernel for {self.value} is not yet implemented.'
            )
        return WARMUP_KERNELS[self.value]


class Scheduler(BaseStrEnum):
    """Learning Rate Scheduler Names."""

    COSINE = 'Cosine'
    LINEAR = 'Linear'

    def get_scheduler(self):
        """Get the learning rate scheduler."""
        from module_sandbox.training.lr_scheduler import (
            cosine_annealing_scheduler,
            linear_decay_scheduler,
        )

        if self.value == 'Cosine':
            return cosine_annealing_scheduler
        if self.value == 'Linear':
            return linear_decay_scheduler
        raise NotImplementedError(
            f'Learning Rate Scheduler for {self.value} is not yet implemented.'
        )


@dataclass(frozen=True)
class PriorConfig(BaseConfig):
    """Configuration for the prior distribution on the model parameters.

    Notes
    -----
    - The `name` should be a `PriorDist` enum value which defines the complete
        prior distribution, it can be a general distribution or a pre-defined one.
        To extend the possible priors, add a new value to the `PriorDist` enum.
        and extend the `get_prior` method accordingly. Through `parameters` field
        the user can pass as many keyword arguments from the configuration file
        as needed for the initialization of the prior distribution.
    """

    name: PriorDist = field(
        default=PriorDist.StandardNormal,
        metadata={'description': 'Prior to Use', 'searchable': True},
    )
    parameters: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            'description': 'Parameters for the prior distribution.',
            'searchable': True,
        },
    )

    def get_prior(self):
        """Get the prior distribution.

        Notes
        -----
        - Get the prior by passing the parameters from the config to `get_prior`
        method of the `PriorDist` enum. See the `PriorDist` enum for more details.
        """
        return self.name.get_prior(**self.parameters)


@dataclass(frozen=True)
class SchedulerConfig(BaseConfig):
    """Scheduler Configuration."""

    name: Scheduler | None = field(
        default=Scheduler.COSINE,
        metadata={'description': 'Scheduler to Use.', 'searchable': True},
    )
    exploration: float = field(
        default=0.25,
        metadata={'description': 'Exploration Ratio.', 'searchable': True},
    )
    target_lr: float = field(
        default=0.0,
        metadata={'description': 'Target Learning Rate.', 'searchable': True},
    )
    n_cycles: int = field(
        default=4,
        metadata={
            'description': 'Number of Cycles [Cosine Scheduler].',
            'searchable': True,
        },
    )

    def __post_init__(self):
        """Post Initialization for the Scheduler Configuration."""
        super().__post_init__()
        n_cycles_default = self.__class__.__dataclass_fields__['n_cycles'].default
        if self.name == Scheduler.LINEAR and self.n_cycles != n_cycles_default:
            self._modify_field(**{'n_cycles': n_cycles_default})
            warnings.warn('Ignoring n_cycles in Linear Scheduler.', UserWarning)

    def get_scheduler(self, n_steps: int, init_lr: float):
        """Get the learning rate scheduler."""
        if self.name == Scheduler.COSINE:
            return self.name.get_scheduler()(
                n_steps=n_steps,
                n_cycles=self.n_cycles,
                init_lr=init_lr,
                target_lr=self.target_lr,
                exploration_ratio=self.exploration,
            )
        if self.name == Scheduler.LINEAR:
            return self.name.get_scheduler()(
                n_steps=n_steps,
                init_lr=init_lr,
                target_lr=self.target_lr,
                exploration_ratio=self.exploration,
            )


@dataclass(frozen=True)
class SamplerConfig(BaseConfig):
    """Sampler Configuration."""

    name: Sampler = field(
        default=Sampler.NUTS, metadata={'description': 'Sampler to Use.'}
    )
    epoch_wise_sampling: bool = field(
        default=False,
        metadata={
            'description': 'Perform epoch-wise or batch-wise in minibatch sampling.'
        },
    )
    params_frozen: list[str] = field(
        default_factory=list,
        metadata={
            'description': (
                'Point delimited parameter names in pytree to freeze.'
                '(not yet fully implemented)'
            )
        },
    )
    batch_size: int | None = field(
        default=None,
        metadata={'description': 'Batch Size in SBI Training.', 'searchable': True},
    )
    warmup_steps: int = field(
        default=50,
        metadata={'description': 'Number of warmup steps.', 'searchable': True},
    )
    n_chains: int = field(
        default=2,
        metadata={'description': 'Number of chains to run.', 'searchable': True},
    )
    n_samples: int = field(
        default=1000,
        metadata={'description': 'Number of samples to draw.', 'searchable': True},
    )
    use_warmup_as_init: bool = field(
        default=True,
        metadata={
            'description': 'Use params resulting from warmup as initial for sampling.'
        },
    )
    n_thinning: int = field(
        default=1, metadata={'description': 'Thinning.', 'searchable': True}
    )
    diagonal_preconditioning: bool = field(
        default=True,
        metadata={
            'description': 'Use Diagonal Preconditioning (MCLMC).',
            'searchable': True,
        },
    )
    desired_energy_var: float = field(
        default=5e-4,
        metadata={
            'description': 'Desired Energy Variance (MCLMC).',
            'searchable': True,
        },
    )
    trust_in_estimate: float = field(
        default=1.5,
        metadata={'description': 'Trust in Estimate (MCLMC).', 'searchable': True},
    )
    num_effective_samples: int = field(
        default=100,
        metadata={
            'description': 'Number of Effective Samples (MCLMC).',
            'searchable': True,
        },
    )
    step_size: float = field(
        default=0.0001, metadata={'description': 'Step Size.', 'searchable': True}
    )
    mdecay: float = field(
        default=0.05, metadata={'description': 'Momentum Decay.', 'searchable': True}
    )
    n_integration_steps: int = field(
        default=1,
        metadata={'description': 'Number of Integration Steps.', 'searchable': True},
    )
    momentum_resampling: float = field(
        default=0.0,
        metadata={'description': 'Momentum Resampling (adaSGHMC)', 'searchable': True},
    )
    temperature: float = field(
        default=1.0, metadata={'description': 'Temperature (SGLD)', 'searchable': True}
    )

    keep_warmup: bool = field(
        default=False, metadata={'description': 'Keep warmup samples.'}
    )
    prior_config: PriorConfig = field(
        default_factory=PriorConfig,
        metadata={'description': 'Prior configuration for the model.'},
    )
    scheduler_config: SchedulerConfig = field(
        default_factory=SchedulerConfig,
        metadata={'description': 'Learning Rate Scheduler Configuration'},
    )

    @property
    def scheduler(self):
        """Get the learning rate scheduler."""
        return self.scheduler_config.get_scheduler(
            n_steps=self.n_samples, init_lr=self.step_size
        )

    def __post_init__(self):
        """Post Initialization for the Sampler Configuration."""
        super().__post_init__()
        mini_batch_only = ['batch_size', 'n_integration_steps', 'mdecay', 'step_size']
        if self.name == Sampler.NUTS:
            for fn in mini_batch_only:
                if getattr(self, fn) is not None:
                    default = self.__class__.__dataclass_fields__[fn].default
                    warnings.warn(f'Ignoring {fn} in NUTS Sampling.', UserWarning)
                    self._modify_field(**{fn: default})

    @property
    def prior(self):
        """Get the prior."""
        return self.prior_config.get_prior()

    @property
    def kernel(self):
        """Returns the kernel: see module_sandbox.training.kernels for more details."""
        return self.name.get_kernel()

    @property
    def warmup_kernel(self):
        """Returns the warmup kernel: see module_sandbox.training.kernels."""
        return self.name.get_warmup_kernel()

    @property
    def _warmup_dir_name(self):
        """Return the directory name for saving warmup samples."""
        return 'sampling_warmup'

    @property
    def _dir_name(self):
        """Return the directory name for saving samples."""
        return 'samples'
