# Configuration Framework for Probabilistic ML

## Description
In this sub-module we provide a framework for defining and managing configuration parameters for probabilistic ML models. The framework is designed to be modular and extensible, allowing for easy integration and customization of configuration parameters. The simplest way to use/explore the framework is to follow following workflow:

## Usage
**Note:** *You can also follow the interactive ipython notebook [here](../../ipynb/config.ipynb)*

We start by importing the top level Config class from the `module_sandbox.config` module.
```python
from module_sandbox.config.core import Config
```

1. As a first step one would generate a config template based on the model one wants to use. One can choose from the pre-defined models in the **module_sandbox.models** or create a custom one.

# Using a pre-defined model:
```python
print(Config.list_avaliable_models())
> ['GPT', 'AttentionClassifier', 'LeNet', 'LeNetti', 'FCN', 'ResFCN']
# Let's say we want to use the LeNet model, we can generate a template for it in yaml format.
Config.template_to_yaml('cfg.yaml', 'LeNet') # yaml serialization
# Config.template_to_json('cfg.json', 'LeNet') # json serialization
# Config.to_template('cfg.yaml', 'LeNet') # dictionary serialization
```
Resulting *.yaml* will contain all the configuration that needs to be filled out for bayesian model training, it contains hints, default values and types.

# Creating a custom model:

1. Create a new configuration class that inherits from the base ModelConfig class and implements the desired configuration parameters.

```python
from dataclasses import dataclass, field
from module_sandbox.config.models.base import ModelConfig, Activation

@dataclass(frozen=True)
class LeNetConfig(ModelConfig):
    model: str = 'LeNet'
    activation: Activation = field(default=Activation.SIGMOID)
    use_bias: bool = True
    outdim: int = 10
```

**Note:**
- *ModelConfig subclasses must contain a `model` attribute that specifies the **exact name of the model class that the configuration is for**, in this case `LeNet`. as we name the module below.*
- *fields must be **type-hinted** otherwise it wont be interpreted as a dataclass field on top which the Framework is built and hereby wont work as desired.*


2. Defining the model

```python

import flax.linen as nn
import jax.numpy as jnp

class LeNet(nn.Module):
    config: LeNetConfig
    some_fix_param: int = 10
    some_other_fix_param = nn.Dense(10)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        act = self.config.activation.flax_activation
        x = x.transpose((0, 2, 3, 1))
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name='conv1'
        )(x)
        x = nn.avg_pool(act(x), window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name='conv2'
        )(x)
        x = nn.avg_pool(act(x), window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120, use_bias=self.config.use_bias, name='fc1')(x)
        x = nn.Dense(features=84, use_bias=self.config.use_bias, name='fc2')(act(x))
        x = nn.Dense(features=self.config.outdim, use_bias=self.config.use_bias, name='fc3')(act(x))
        return x
```
**Note:**
- *The model class must have a field `config` holding your custom-defined config subclass.*
- *You can define other parameters that you dont want to be configurable and part of the config.*

3. Generating a template for the custom model.

```python
# Now we pass the actual custom class to the template_to_yaml method.
Config.template_to_yaml('cfg.yaml', LeNet)
```

# Loading the configuration

After filling out the configuration, we can load it back into a Config object and use it for training.
During the loading process, the Config object will validate the types and custom constraints defined for each entry in the configuration.

```python
cfg = Config.from_yaml('cfg.yaml')
# cfg = Config.from_json('cfg.json')
# cfg = Config.from_dict({...})
```
Important method of the Config object is the `.get_flax_model()` method, which will return a Flax `nn.Module` object based on the configuration. in this case a configured ready to use `LeNet` model.

```python
model = cfg.get_flax_model()
```
