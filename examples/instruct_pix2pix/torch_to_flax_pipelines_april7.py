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
import numpy as np 
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
    UNet2DConditionModel,
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
    FlaxEulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
    FlaxStableDiffusionImg2ImgPipeline,
    FlaxStableDiffusionInstructPix2PixPipeline,
  )

from diffusers.utils import make_image_grid
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

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    revision="flax",
    dtype=dtype,
)

# %%

# Save flax pipeline to disk as a local pipeline

outdir = '../flax_models/stable-diffusion-v1-5'

scheduler, _ = FlaxPNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True
)
pipeline = FlaxStableDiffusionPipeline(
    text_encoder=pipeline.text_encoder,
    vae=pipeline.vae,
    unet=pipeline.unet,
    tokenizer=pipeline.tokenizer, # FlaxPreTrainedModel.save_pretrained of PreTrainedTokenizer
    scheduler=scheduler,
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)
# %% 
# NOTE: three possible library classes each have two possible save pretrained methods:
# library_classes
# {'FlaxModelMixin': ['save_pretrained', 'from_pretrained'], 'FlaxSchedulerMixin': ['save_pretrained', 'from_pretrained'], 'FlaxDiffusionPipeline': ['save_pretrained', 'from_pretrained']}
# save_load_methods
# ['save_pretrained', 'from_pretrained']

#  for vae, the save_method is = FlaxModelMixin.save_pretrained , 
#  save_directory =  '../flax_models/stable-diffusion-v1-5', pipeline_component_name = 'vae', expects_params=True, 
#  which is called here:
# if expects_params:
# save_method( # os.path.join(save_directory, pipeline_component_name), params=params[pipeline_component_name])


pipeline.save_pretrained(
    outdir,
    params={
        "text_encoder": params["text_encoder"],  # FlaxPreTrainedModel.save_pretrained
        "vae": params["vae"],
        "unet": params["unet"],
        "safety_checker": safety_checker.params,
    },
)




# %%
#                                                                Pytorch weights
################################################################################


dtype = jnp.bfloat16

from diffusers import FlaxEulerDiscreteScheduler
#scheduler, _ = FlaxEulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

scheduler, scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    subfolder="scheduler",

)

pipeline, params  = FlaxStableDiffusionPipeline.from_pretrained(
    'timbrooks/instruct-pix2pix',
    scheduler=scheduler, # over
    from_pt=True,
)

outdir = '../gatech/instruct-pix2pix'

safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True
)

# %% 

pipeline = FlaxStableDiffusionPipeline(
    text_encoder=pipeline.text_encoder,
    vae=pipeline.vae,
    unet=pipeline.unet,
    tokenizer=pipeline.tokenizer, # FlaxPreTrainedModel.save_pretrained of PreTrainedTokenizer
    scheduler=scheduler, # FlaxSchedulerMixin.save_pretrained of FlaxEulerDiscreteScheduler 
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)
# %% 


pipeline.save_pretrained(
    outdir,
    params={
        "text_encoder": params["text_encoder"],  # FlaxPreTrainedModel.save_pretrained
        "vae": params["vae"],
        "unet": params["unet"],
        "safety_checker": safety_checker.params,
    },
)



# %% 
#                                       Convert and Download Instruct-Pix2Pix Pipeline
######################################################################################


# Load models and create wrapper for stable diffusion
tokenizer = CLIPTokenizer.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    revision="flax",
    subfolder="tokenizer",
)

text_encoder = FlaxCLIPTextModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    revision="flax",
    subfolder="text_encoder",
)
vae, vae_state = FlaxAutoencoderKL.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    revision="flax",
    subfolder="vae",
)

scheduler, scheduler_state = FlaxEulerDiscreteScheduler.from_pretrained(
    'timbrooks/instruct-pix2pix',
    from_pt=True,
    revision='main',
    subfolder='scheduler',
)

unet, unet_state = FlaxUNet2DConditionModel.from_pretrained(
    'timbrooks/instruct-pix2pix',
    from_pt=True,
    revision='main',
    subfolder='unet',
)

safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True
)

pipeline = FlaxStableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer, # FlaxPreTrainedModel.save_pretrained of PreTrainedTokenizer
    scheduler=scheduler, # FlaxSchedulerMixin.save_pretrained of FlaxEulerDiscreteScheduler 
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)

# %% 

text_encoder_state = text_encoder._params

pipeline.save_pretrained(
    '../flax_models/instruct-pix2pix',
    params={
        "text_encoder": text_encoder_state,  # FlaxPreTrainedModel.save_pretrained
        "vae": vae_state,
        "unet": unet_state,
        "safety_checker": safety_checker.params,
    },
)
####################################################################################################################################

# %%

dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="flax",
    dtype=dtype,
)

# %%

# Save flax pipeline to disk as a local pipeline

outdir = '../flax_models/stable-diffusion-v1-5'

scheduler, _ = FlaxPNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True
)
pipeline = FlaxStableDiffusionPipeline(
    text_encoder=pipeline.text_encoder,
    vae=pipeline.vae,
    unet=pipeline.unet,
    tokenizer=pipeline.tokenizer, # FlaxPreTrainedModel.save_pretrained of PreTrainedTokenizer
    scheduler=scheduler,
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)
# %% 

pipeline.save_pretrained(
    outdir,
    params={
        "text_encoder": params["text_encoder"],  # FlaxPreTrainedModel.save_pretrained
        "vae": params["vae"],
        "unet": params["unet"],
        "safety_checker": safety_checker.params,
    },
)


################################################################################

# %%
                                                                # Flax inference
################################################################################

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
  # '../flax_models/stable-diffusion-v1-5',
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



# %%
                                                                # Flax inference
################################################################################

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




