# %% 

# Load a modified pretrained PyTorch model, create a new Flax model with the same architecture, and initialize it with the modified weights
# Finally, save the modified Flax model with the updated weights

import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from diffusers import FlaxUNet2DConditionModel

model_id = 'runwayml/stable-diffusion-v1-5'
unet = FlaxUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", revision='flax', in_channels=8, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)



unet, state = FlaxUNet2DConditionModel.from_pretrained(
  'runwayml/stable-diffusion-v1-5',
  revision='flax',
  subfolder="unet",
  dtype=jnp.bfloat16,
  )

in_channels = state['conv_in']['kernel']

# Initialize new weights for extra channels
kernel_shape = state['conv_in']['kernel'].shape
bias_shape = state['conv_in']['bias'].shape

new_channels = jnp.zeros(kernel_shape, dtype=jnp.float32)

# Concatenate along the input channel dimension
new_conv_weights = jnp.concatenate([in_channels, new_channels], axis=2)

# Update the weights dictionary
state['conv_in']['kernel'] = new_conv_weights


        # init input tensors
#         sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
#         sample = jnp.zeros(sample_shape, dtype=jnp.float32)
#         timesteps = jnp.ones((1,), dtype=jnp.int32)
#         encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)

# update the model's in_channels attribute
unet.in_channels = 8

# Initialize the model to create the parameter structure
sample_shape = (1, unet.in_channels, unet.sample_size, unet.sample_size)
sample = jnp.zeros(sample_shape, dtype=jnp.float32)
timesteps = jnp.ones((1,), dtype=jnp.int32)
encoder_hidden_states = jnp.zeros((1, 1, unet.cross_attention_dim), dtype=jnp.float32)
variables = unet.init(jax.random.PRNGKey(0), sample, timesteps, encoder_hidden_states)


# Now replace the initialized weights with your adjusted weights
variables = unfreeze(variables)
variables['params'].update(state)
variables = freeze(variables)
# %%

import os 

 # save the modified unet and updated parameters
unet.save_pretrained('modified_unet', params=variables['params']) 


# %%
# Initialize and load your Flax model from the modified weights
sample_shape = (1, unet.in_channels, unet.sample_size, unet.sample_size)
sample = jnp.zeros(sample_shape, dtype=jnp.float32)
timesteps = jnp.ones((1,), dtype=jnp.int32)
encoder_hidden_states = jnp.zeros((1, 1, unet.cross_attention_dim), dtype=jnp.float32)
state = unet.init(jax.random.PRNGKey(0), sample, timesteps, encoder_hidden_states)

diffuser_model = FlaxUNet2DConditionModel(unet)
diffuser_model.model_variables = state
