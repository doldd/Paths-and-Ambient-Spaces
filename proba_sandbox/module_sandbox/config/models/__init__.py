"""Configuration for deep learning models."""
from proba_sandbox.module_sandbox.config.models.cnns import LeNetConfig, LeNettiConfig
from proba_sandbox.module_sandbox.config.models.fcn import FCNConfig, ResFCNConfig
from proba_sandbox.module_sandbox.config.models.gpt import GPTConfig

__all__ = [
    'GPTConfig',
    'FCNConfig',
    'ResFCNConfig',
    'LeNetConfig',
    'LeNettiConfig',
]
