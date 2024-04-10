# %% 
import os
import time
import warnings
from functools import partial
from typing import Dict, List, Optional, Union

import jax
import numpy as np
import jax.numpy as jnp
from jax import pmap
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from flax.jax_utils import replicate
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard
import requests
from io import BytesIO
from PIL import Image

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

from diffusers.utils import  PIL_INTERPOLATION, logging, replace_example_docstring
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionPipelineOutput



# local imports
from flax_from_hf_pretrained_april4 import get_pretrained # converts Torch model to Flax

unet, unet_params           = get_pretrained("timbrooks/instruct-pix2pix", 'unet', FlaxUNet2DConditionModel)
vae, vae_params             = get_pretrained("timbrooks/instruct-pix2pix", 'vae', FlaxAutoencoderKL)
scheduler, schduler_params  = FlaxDDPMScheduler.from_pretrained("timbrooks/instruct-pix2pix", subfolder="scheduler")
text_encoder                = FlaxCLIPTextModel.from_pretrained("timbrooks/instruct-pix2pix", subfolder='text_encoder', dtype=jnp.bfloat16)
tokenizer                   = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder='tokenizer', dtype=jnp.bfloat16)

# %%
dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained(
  '../flax_models/instruct-pix2pix',
  dtype=dtype,
  revision='flax',
  tokenizer=tokenizer,
  text_encoder=text_encoder,
  vae=vae,
  unet=unet,
  scheduler=scheduler,
  )
  


# %%

# create a dummy pipeline and params object

params = {}
pipeline = {}

params['unet'] = unet_params
params['vae'] = vae_params
params['scheduler'] = schduler_params
params['text_encoder'] = text_encoder.params

pipeline['unet'] = unet
pipeline['vae'] = vae
pipeline['scheduler'] = scheduler
pipeline['text_encoder'] = text_encoder
pipeline['tokenizer'] = tokenizer

params = FrozenDict(params)

# %% 
def pipeline_prepare_inputs( prompt: Union[str, List[str]], image: Union[Image.Image, List[Image.Image]]):
    if not isinstance(prompt, (str, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if not isinstance(image, (Image.Image, list)):
        raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

    if isinstance(image, Image.Image):
        image = [image]

    processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    return text_input.input_ids, processed_images


def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)


def preprocess(image, dtype):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return 2.0 * image - 1.0


def create_key(seed=0):
    return jax.random.PRNGKey(seed)


# %% 

_pipeline, _params = FlaxStableDiffusionInstructPix2PixPipeline.from_pretrained('../flax_models/instruct-pix2pix', revision='flax', dtype=jnp.bfloat16)

# %%

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
image = init_img.resize((768, 512))
prompt = "A fantasy landscape, trending on artstation"


# 2. We cast all parameters to bfloat16 EXCEPT the scheduler which we leave in
# float32 to keep maximal precision
params = unfreeze(params)
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)

params["scheduler"] = scheduler_state

params = freeze(params)

# 3. Next, we define the different inputs to the pipeline
default_prompt = prompt 
default_neg_prompt = ""
default_image = image 
seed = 33
default_guidance_scale = 5.0
default_num_steps = 25
width = 1024
height = 1024

NUM_DEVICES = jax.device_count()

# num_samples = jax.device_count()
rng = jax.random.PRNGKey(seed)
rng = jax.random.split(rng, jax.device_count())

p_params = replicate(params)



p_params = replicate(params)

prompt_ids, image_ids = pipeline_prepare_inputs(prompt=[prompt] * NUM_DEVICES, image=[image] * NUM_DEVICES)
prompt_ids = shard(prompt_ids)
image_ids = shard(image_ids)



output = FlaxStableDiffusionInstructPix2PixPipeline.__call__(
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


# %%


def replicate_all(prompt_ids: jnp.ndarray, neg_prompt_ids: jnp.ndarray, image_ids: jnp.ndarray, seed: int):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    p_image_ids = replicate(image_ids)

    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, p_image_ids, rng


def prepare_inputs(prompt: str, negative_prompt: str, image: Image.Image):
    prompt_ids, image_ids = pipeline_prepare_inputs(prompt=[prompt] , image=[image]) 
    neg_prompt_ids, _ = pipeline_prepare_inputs(prompt=[negative_prompt], image=[image]) 
    # prompt_ids, neg_prompt_ids, image_ids = replicate_all(prompt_ids, neg_prompt_ids, image_ids, seed)
    return prompt_ids, neg_prompt_ids, image_ids


# %%
    
def aot_compile(
        prompt=default_prompt,
        image=default_image,
        seed=seed,
        guidance_scale=default_guidance_scale,
        num_inference_steps=default_num_steps
):
    # prompt_ids, neg_prompt_ids, image_ids = prepare_inputs(prompt, negative_prompt, image)
    prompt_ids, image_ids = pipeline_prepare_inputs(prompt=[prompt] * NUM_DEVICES, image=[image] * NUM_DEVICES)

    prompt_ids = shard(prompt_ids)
    image_ids = shard(image_ids)

    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    guidance_scale = g[:, None]

    noise = None

    return pmap(
        FlaxStableDiffusionInstructPix2PixPipeline._generate,static_broadcasted_argnums=[3, 4, 5, 9]
        ).lower(
            prompt_ids,           # jnp.ndarray,
            image_ids,            # jnp.ndarray,
            p_params,             # FrozenDict,
            rng,                  # jax.Array,
            num_inference_steps,  # int (num_inference_steps)
            height,               # Optional[int]
            width,                # Optional[int]
            guidance_scale,       # Union[float, jnp.ndarray]
            None,                 # Optional[jnp.ndarray] (noise)
            False,                # bool (return_latents)
            ).compile()


start = time.time()
print("Compiling ...")
p_generate = aot_compile()
print(f"Compiled in {time.time() - start}")

# %% 
            
def generate(
    prompt,
    negative_prompt,
    seed=seed,
    guidance_scale=default_guidance_scale
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    guidance_scale  = g[:, None]
    images = p_generate(
        prompt_ids,
        p_params,
        rng,
        guidance_scale,
        None,
        neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))            



# %%


start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"First inference in {time.time() - start}")