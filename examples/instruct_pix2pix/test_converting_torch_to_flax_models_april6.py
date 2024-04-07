# %%

import os
import sys
import time
import requests
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt
from pickle import UnpicklingError
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
from jax import pmap
# Let's cache the model compilation, so that it doesn't take as long the next time around.
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("/tmp/sdxl_cache")

NUM_DEVICES = jax.device_count()

from flax.jax_utils import replicate
from flax.training.common_utils import shard
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict

import torch
from huggingface_hub import create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)
from transformers import FlaxCLIPTextModel, CLIPTokenizer

from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    WEIGHTS_NAME,
    PushToHubMixin,
)

from transformers import (
    CLIPImageProcessor,
    CLIPTokenizer, 
    FlaxCLIPTextModel
    )
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
    FlaxStableDiffusionImg2ImgPipeline,
    FlaxStableDiffusionInstructPix2PixPipeline,
  )

from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from diffusers.models.modeling_utils import load_state_dict


# local imports
from flax_from_hf_pretrained_april4 import get_pretrained
from model_converter import load_from_standard_weights


#%%

dtype = jnp.bfloat16
MODEL_NAME="CompVis/stable-diffusion-v1-4",
MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
MODEL_NAME="runwayml/stable-diffusion-v1-5"

from diffusers import DiffusionPipeline
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    revision="flax",
    dtype=dtype,
)



# %%
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image
from diffusers import FlaxStableDiffusionImg2ImgPipeline

def create_key(seed=0):
    return jax.random.PRNGKey(seed)


rng = create_key(0)

dtype = jnp.bfloat16
# load the pipeline
pipeline, params  = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="flax",
    dtype=dtype,
)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompts = "A fantasy landscape, trending on artstation"

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image = pipeline.prepare_inputs(
    prompt=[prompts] * num_samples, image=[init_image] * num_samples
)
p_params = replicate(params)
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)

output = pipeline(
    prompt_ids=prompt_ids,
    image=processed_image,
    params=p_params,
    prng_seed=rng,
    strength=0.75,
    num_inference_steps=50,
    jit=True,
    height=512,
    width=768,
).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
# output_images[0].save("fantasy_landscape.png")

# %%


# %%





# %%

import torch

# load the pipeline
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16,
# )
scheduler, schduler_params = FlaxDDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", dtype=jnp.bfloat16)
text_encoder = FlaxCLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='text_encoder', dtype=jnp.bfloat16)
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='tokenizer', dtype=jnp.bfloat16)
vae, vae_params = get_pretrained("runwayml/stable-diffusion-v1-5", 'vae', FlaxAutoencoderKL)
unet, unet_params = get_pretrained("runwayml/stable-diffusion-v1-5", 'unet', FlaxUNet2DConditionModel)

outdir = '../flax_models/stable-diffusion-v1-5'
scheduler, _ = FlaxPNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True
)

pipeline = FlaxStableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)
# %% 

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


pipeline.save_pretrained(
    outdir,
    params={
        "text_encoder": None, 
        "vae": vae_params, 
        "unet": unet_params,
        "safety_checker": safety_checker.params,
    },
)




# %%





# %%


# flax model saved to disk from flax pipeline (working)

outdir = '../flax_models/stable-diffusion-v1-5'

scheduler, _ = FlaxPNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True
)
pipeline = FlaxStableDiffusionPipeline(
    text_encoder=pipeline.text_encoder,
    vae=pipeline.vae,
    unet=pipeline.unet,
    tokenizer=pipeline.tokenizer,
    scheduler=scheduler,
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)

pipeline.save_pretrained(
    outdir,
    params={
        "text_encoder": params["text_encoder"],
        "vae": params["vae"],
        "unet": params["unet"],
        "safety_checker": safety_checker.params,
    },
)




# %%

import jax
import jax.numpy as jnp
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline
def create_key(seed=0):
    return jax.random.PRNGKey(seed)

dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
  '../flax_models/stable-diffusion-v1-5',
  revision="flax",
  dtype=dtype,
)

prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape
# (8, 77)

# parameters
p_params = replicate(params)

# arrays
prompt_ids = shard(prompt_ids)
prompt_ids.shape
# (8, 1, 77)

rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())

images = pipeline(prompt_ids, p_params, rng, jit=True)[0]

# CPU times: user 56.2 s, sys: 42.5 s, total: 1min 38s
# Wall time: 1min 29s

from diffusers.utils import make_image_grid

images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
make_image_grid(images, rows=1, cols=4)
# %%
