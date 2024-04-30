# %%

import tempfile
from io import BytesIO

import jax
import jax.numpy as jnp
import numpy as np
import requests
import torch
from diffusers import FlaxUNet2DConditionModel, UNet2DConditionModel
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

# Download EMA only checkpoints
# NOTE: Flax models must use revision=`bfloat16` to load the EMA only checkpoints.
# NOTE: PyTorch models use revision=`fp16` to load the EMA only checkpoints.
# NOTE: revision and dtype are not the same thing even thought they may be the same value.

# `raw` weights (not EMA smoothed) use revision=`flax` for Flax models and variant=`non_ema` or revision=`non_ema` for PyTorch models.(or `main` and a `variant` specifier)

base_model = UNet2DConditionModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    sample_size=32,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32,
)

instruct_model = UNet2DConditionModel.from_pretrained(
    "timbrooks/instruct-pix2pix",
    subfolder="unet",  # NOTE: for instruct-pix2pix, variants are `main` and `fp16` and correspond to precision not EMA weights (here we load main or fp32 precision weights by default)
)

# Creates a PyTorch unet model with 8 input channels instead of the default 4, then loads the v1-5-pruned-emaonly.ckpt using the `fp16` PyTorch revision.

model = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    in_channels=8,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    variant="fp16",  # NOTE: `fp16` is used to load the EMA-only PyTorch weights, variant=`non_ema` for the raw weights
)

# NOTE: FutureWarning: You are loading the variant fp16 from runwayml/stable-diffusion-v1-5 via `revision='fp16'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='fp16'` instead. However, it appears that runwayml/stable-diffusion-v1-5 currently does not have a diffusion_pytorch_model.fp16.safetensors file in the 'main' branch of runwayml/stable-diffusion-v1-5.


model = model.eval()
instruct_model = instruct_model.eval()

with tempfile.TemporaryDirectory() as tmpdirname:
    # model.save_pretrained(tmpdirname)
    model.save_pretrained(
        tmpdirname, safe_serialization=False
    )  # safe_serialization=False is needed to save the model as a .bin file and not a safetensors file
    flax_model, flax_params = FlaxUNet2DConditionModel.from_pretrained(
        tmpdirname, from_pt=True
    )

sample = torch.rand(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)
time = 1
text_emb = torch.rand(1, 1, model.config.cross_attention_dim)

print(f"Input shape: {sample.shape}")

# Step 1: Infer with the PT model
torch_output = model(sample, time, text_emb).sample

instruct_torch_output = instruct_model(sample, time, text_emb).sample

# Check that the outputs have the same shape
assert torch_output.shape == instruct_torch_output.shape

# Step 2: Infer with JAX model
flax_sample = jnp.array(sample.numpy())
flax_text_emb = jnp.array(text_emb.numpy())

flax_output = flax_model.apply(
    {"params": flax_params}, flax_sample, time, flax_text_emb
).sample

# Step 3: Check that the values are close
converted_flax_output = torch.from_numpy(np.array(flax_output))

torch.testing.assert_close(converted_flax_output, torch_output, rtol=4e-03, atol=4e-03)

import json
import os

# Define the directory where the model and its configuration will be saved
save_dir = "modified_unet"

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the model parameters
params_path = os.path.join(save_dir, "flax_model_params.msgpack")
flax_model.save_pretrained(save_dir, params=flax_params, _internal_call=True)

# %%

assert flax_model.dtype == jax.numpy.float32  # EMA-only weights in float32

# NOTE: these weights would correspond to HF's variant='bfloat16' (Hugging Face convention for Flax EMA-only weights) even though the model precision is not bfloat16

# Save the model configuration
config_path = os.path.join(save_dir, "config.json")

with open(config_path, "w") as f:
    # json.dump(unfreeze(flax_model.config).to_dict(), f)
    json.dump(flax_model.config, f)

print(f"Model, parameters, and configuration saved in directory: {save_dir}")


del flax_model, flax_params

# Load the model and its parameters

save_dir = "modified_unet"
flax_model, flax_params = FlaxUNet2DConditionModel.from_pretrained(save_dir)

# Check that the model and parameters have been loaded correctly

assert flax_model.config.in_channels == 8
assert flax_model.config.block_out_channels == [320, 640, 1280, 1280]
