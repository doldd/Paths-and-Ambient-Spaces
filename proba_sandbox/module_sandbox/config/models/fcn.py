"""Fully connected NN model configuration."""
from dataclasses import dataclass, field

from proba_sandbox.module_sandbox.config.models.base import Activation, ModelConfig


@dataclass(frozen=True)
class FCNConfig(ModelConfig):
    """FCN Model Configuration."""

    model: str = 'FCN'
    hidden_structure: list[int] = field(
        hash=False,
        default_factory=lambda: [10, 10],
        metadata={
            'description': 'Width (+ number) of hidden layers',
            'searchable': True,
        },
    )
    activation: Activation = field(
        default=Activation.RELU,
        metadata={
            'description': 'Activation func. for hidden layers',
            'searchable': True,
        },
    )

    use_bias: bool = field(
        default=True, metadata={'description': 'Whether to include bias terms'}
    )


@dataclass(frozen=True)
class ResFCNConfig(ModelConfig):
    """Residual FCN Model Configuration."""

    model = 'ResFCN'
    n_blocks: int = field(
        default=3, metadata={'description': 'Number of residual blocks'}
    )
    hidden_block_structure: list[int] = field(
        hash=False,
        default_factory=lambda: [10, 10],
        metadata={'description': 'Width (+ number) of hidden layers'},
    )
    use_bias: bool = field(
        default=True, metadata={'description': 'Whether to include bias terms'}
    )
    last_layer_structure: int = field(
        default=2, metadata={'description': 'Width of the last layer (output)'}
    )
    activation: Activation = field(
        default=Activation.RELU,
        metadata={'description': 'Activation func. for hidden layers'},
    )
