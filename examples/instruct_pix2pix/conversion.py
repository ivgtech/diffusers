# %% 
from diffusers import UNet2DConditionModel, FlaxUNet2DConditionModel
import tempfile
import torch

import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline
from diffusers import FlaxStableDiffusionInstructPix2PixPipeline
from diffusers import UNet2DConditionModel, FlaxUNet2DConditionModel

from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
 

base_model = UNet2DConditionModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    sample_size=32,
    in_channels=4,
    out_channels=4,
    down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
    up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
    cross_attention_dim=32,
)

instruct_model = UNet2DConditionModel.from_pretrained(
    'timbrooks/instruct-pix2pix',
    subfolder='unet',
)

# Creates a PyTorch unet model with 8 input channels instead of the default 4, then loads non-EMA weights from the 'flax' revision
model = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    subfolder='unet',
    # revision='non-ema', # Using a non-documented approach to load the non-EMA weights via the 'flax' revision
    in_channels=8,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
)

model = model.eval()
instruct_model = instruct_model.eval()
# %% 
with tempfile.TemporaryDirectory() as tmpdirname:
    # model.save_pretrained(tmpdirname)
    model.save_pretrained(tmpdirname, safe_serialization=False) # safe_serialization=False is needed to save the model as a .bin file and not a safetensors file
    flax_model, flax_params = FlaxUNet2DConditionModel.from_pretrained(tmpdirname, from_pt=True)

sample = torch.rand(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
time = 1
text_emb = torch.rand(1, 1, model.config.cross_attention_dim)


# %% 
# Step 1: Infer with the PT model
torch_output = model(sample, time, text_emb).sample

instruct_torch_output = instruct_model(sample, time, text_emb).sample

# Check that the outputs have the same shape
assert torch_output.shape == instruct_torch_output.shape

# Step 2: Infer with JAX model
flax_sample = jnp.array(sample.numpy())
flax_text_emb = jnp.array(text_emb.numpy())

flax_output = flax_model.apply({'params':flax_params}, flax_sample, time, flax_text_emb).sample

# Step 3: Check that the values are close
converted_flax_output = torch.from_numpy(np.array(flax_output))

torch.testing.assert_close(converted_flax_output, torch_output, rtol=4e-03, atol=4e-03)
# %%

import os
import json

# Define the directory where the model and its configuration will be saved
save_dir = 'modified_unet'

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the model parameters
params_path = os.path.join(save_dir, 'flax_model_params.msgpack')
flax_model.save_pretrained(save_dir, params=flax_params, _internal_call=True)

# Save the model configuration
config_path = os.path.join(save_dir, 'config.json')

with open(config_path, 'w') as f:
    # json.dump(unfreeze(flax_model.config).to_dict(), f)
    json.dump(flax_model.config, f)

print(f'Model, parameters, and configuration saved in directory: {save_dir}')



# %%

del flax_model, flax_params

# Load the model and its parameters

save_dir = 'modified_unet'
flax_model, flax_params = FlaxUNet2DConditionModel.from_pretrained(save_dir)

# Check that the model and parameters have been loaded correctly

assert flax_model.config.in_channels == 8
assert flax_model.config.block_out_channels == [320, 640, 1280, 1280]

# %%

