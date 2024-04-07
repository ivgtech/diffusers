# %%

import os
import sys
import time
import requests
from io import BytesIO

import warnings
from functools import partial
from typing import Dict, List, Optional, Union
from packaging import version

from PIL import Image
import matplotlib.pyplot as plt
from pickle import UnpicklingError
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np 
from jax import pmap
# Let's cache the model compilation, so that it doesn't take as long the next time around.
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("/tmp/sdxl_cache")

NUM_DEVICES = jax.device_count()

from flax.jax_utils import replicate, unreplicate
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
    FlaxDiffusionPipeline,
  )

from diffusers.utils import make_image_grid
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from diffusers.models.modeling_utils import load_state_dict
from diffusers.utils import PIL_INTERPOLATION

from diffusers.utils import deprecate, logging, replace_example_docstring
from diffusers.pipelines import FlaxDiffusionPipeline
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker_flax import FlaxStableDiffusionSafetyChecker

# local imports
from flax_from_hf_pretrained_april4 import get_pretrained
from model_converter import load_from_standard_weights

DEBUG = False

#%%

# %%



DEBUG = False 

class FlaxSD(FlaxDiffusionPipeline):

    def __init__(
        self,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxCLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        safety_checker: FlaxStableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()

        self.dtype = dtype

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def get_timestep_start(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)

        return t_start

    def _generate(
        self,
        prompt_ids: jnp.ndarray,
        image: jnp.ndarray,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        start_timestep: int,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        noise: Optional[jnp.ndarray] = None,
        neg_prompt_ids: Optional[jnp.ndarray] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        print("_generate func", prompt_ids.shape)
        # get prompt text embeddings
        prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = prompt_ids.shape[0]
        max_length = prompt_ids.shape[-1]

        if neg_prompt_ids is None:
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
            ).input_ids
        else:
            uncond_input = neg_prompt_ids

        negative_prompt_embeds = self.text_encoder(uncond_input, params=params["text_encoder"])[0]
        context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])

        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if noise is None:
            noise = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
        else:
            if noise.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {noise.shape}, expected {latents_shape}")

        # Create init_latents
        init_latent_dist = self.vae.apply({"params": params["vae"]}, image, method=self.vae.encode).latent_dist
        init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))
        init_latents = self.vae.config.scaling_factor * init_latents

        def loop_body(step, args):

            latents, scheduler_state = args
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latents_input = jnp.concatenate([latents] * 2)

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            latents_input = self.scheduler.scale_model_input(scheduler_state, latents_input, t)

            # predict the noise residual
            noise_pred = self.unet.apply(
                {"params": params["unet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=context,
            ).sample
            # perform guidance
            noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            return latents, scheduler_state

        # set the scheduler state
        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
        )

        latent_timestep = scheduler_state.timesteps[start_timestep : start_timestep + 1].repeat(batch_size)
        latents = self.scheduler.add_noise(params["scheduler"], init_latents, noise, latent_timestep)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma
        if DEBUG:
            # run with python for loop
            for i in range(start_timestep, num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(start_timestep, num_inference_steps, loop_body, (latents, scheduler_state))

        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image

    def __call__(
        self,
        prompt_ids: jnp.ndarray,
        image: jnp.ndarray,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Union[float, jnp.ndarray] = 7.5,
        noise: jnp.ndarray = None,
        neg_prompt_ids: jnp.ndarray = None,
        return_dict: bool = True,
        jit: bool = False,
    ):
    
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]

        print(prompt_ids.shape)
        start_timestep = self.get_timestep_start(num_inference_steps, strength)
        if jit:
            images = _p_generate( self, prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,)
        else:
            images = self._generate( prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,)

        images = np.asarray(images)
        has_nsfw_concept = False
        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)





# Static argnums are pipe, start_timestep, num_inference_steps, height, width. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 5, 6, 7, 8),
)
def _p_generate(pipe,      prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,):
    return pipe._generate( prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,)




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




# %%


dtype = jnp.bfloat16
# pipeline, params = FlaxSD.from_pretrained(

pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    '../flax_models/stable-diffusion-v1-5',
    revision="flax",
    dtype=dtype,
)




# %%

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

def prepare_inputs(prompt: Union[str, List[str]], image: Union[Image.Image,  List[Image.Image]], negative_prompt: Union[str, List[str], None], tokenizer: CLIPTokenizer):
    if not isinstance(prompt, (str, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if not isinstance(negative_prompt, (str, list)):
        raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

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

    neg_text_input = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )

    return text_input.input_ids, processed_images, neg_text_input.input_ids

def numpy_to_pil(images):
    #  from src/diffusers/pipelines/pipeline_flax_utils.py
    if images.ndim == 3: images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

rng = create_key(0)
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
prompts = "A fantasy landscape, trending on artstation"
neg_prompts = "not a fantasy landscape, trending on artstation" 

num_samples = jax.device_count()
rng = jax.random.split(rng, jax.device_count())
prompt_ids, processed_image, neg_prompt_ids = prepare_inputs(
    prompt=[prompts] * num_samples, image=[init_img] * num_samples,
    negative_prompt=[neg_prompts] * num_samples,
    tokenizer=pipeline.tokenizer,
)

p_params = replicate(params)
p_prompt_ids = shard(prompt_ids)
p_neg_prompt_ids = shard(neg_prompt_ids)
p_processed_image = shard(processed_image)



# %% 


output = pipeline(
    prompt_ids=p_prompt_ids,
    image=p_processed_image,
    params=p_params,
    prng_seed=rng,
    strength=0.75,
    num_inference_steps=50,
    height=512,
    width=768,
    jit=True,
).images

output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
output_images[0].show()


# %% 
DEBUG = False

vae = pipeline.vae
text_encoder = pipeline.text_encoder
tokenizer = pipeline.tokenizer
unet = pipeline.unet
scheduler = pipeline.scheduler
safety_checker = pipeline.safety_checker
feature_extractor = pipeline.feature_extractor
dtype = pipeline.dtype

def my_generate(
    prompt_ids: jnp.ndarray,
    image: jnp.ndarray,
    params: Union[Dict, FrozenDict],
    prng_seed: jax.Array,
    start_timestep: int,
    num_inference_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    noise: Optional[jnp.ndarray] = None,
    neg_prompt_ids: Optional[jnp.ndarray] = None,
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    # get prompt text embeddings
    print("my_generate prompt_ids", prompt_ids.shape, type(unet))
    prompt_embeds = text_encoder(prompt_ids, params=params["text_encoder"])[0]
    print("my_generate prompt_embeds", prompt_embeds.shape)

    # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
    # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
    batch_size = prompt_ids.shape[0]
    max_length = prompt_ids.shape[-1]

    if neg_prompt_ids is None:
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
        ).input_ids
    else:
        uncond_input = neg_prompt_ids

    negative_prompt_embeds = text_encoder(uncond_input, params=params["text_encoder"])[0]
    context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])

    latents_shape = (
        batch_size,
        unet.config.in_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    if noise is None:
        noise = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
    else:
        if noise.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {noise.shape}, expected {latents_shape}")

    # Create init_latents
    init_latent_dist = vae.apply({"params": params["vae"]}, image, method=vae.encode).latent_dist
    init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))
    init_latents = vae.config.scaling_factor * init_latents

    def loop_body(step, args):
        latents, scheduler_state = args
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        latents_input = jnp.concatenate([latents] * 2)

        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        timestep = jnp.broadcast_to(t, latents_input.shape[0])

        latents_input = scheduler.scale_model_input(scheduler_state, latents_input, t)

        # predict the noise residual
        noise_pred = unet.apply(
            {"params": params["unet"]},
            jnp.array(latents_input),
            jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context,
        ).sample

        # perform guidance
        noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
        return latents, scheduler_state

    # set the scheduler state
    scheduler_state = scheduler.set_timesteps(
        params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
    )

    latent_timestep = scheduler_state.timesteps[start_timestep : start_timestep + 1].repeat(batch_size)
    latents = scheduler.add_noise(params["scheduler"], init_latents, noise, latent_timestep)
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * params["scheduler"].init_noise_sigma
    if DEBUG:
        # run with python for loop
        for i in range(start_timestep, num_inference_steps):
            latents, scheduler_state = loop_body(i, (latents, scheduler_state))
    else:
        latents, _ = jax.lax.fori_loop(start_timestep, num_inference_steps, loop_body, (latents, scheduler_state))

    # scale and decode the image latents with vae
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.apply({"params": params["vae"]}, latents, method=vae.decode).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image




# %%
# Call 


# Static argnums are pipe, start_timestep, num_inference_steps, height, width. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(0, 0, 0, 0, None, None, None, None, 0, 0, 0),
    static_broadcasted_argnums=(5, 6, 7, 8),
)
def my_p_generate(      prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,):
    return my_generate( prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,)






def get_timestep_start(num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    return t_start

def my_call(
    prompt_ids: jnp.ndarray,
    image: jnp.ndarray,
    params: Union[Dict, FrozenDict],
    prng_seed: jax.Array,
    strength: float = 0.8,
    num_inference_steps: int = 50,
    height: Optional[int] = None,
    width: Optional[int] = None,
    guidance_scale: Union[float, jnp.ndarray] = 7.5,
    noise: jnp.ndarray = None,
    neg_prompt_ids: jnp.ndarray = None,
    return_dict: bool = True,
    jit: bool = False,
):

    # 0. Default height and width to unet
    height = height or unet.config.sample_size * vae_scale_factor
    width = width or unet.config.sample_size * vae_scale_factor
    if isinstance(guidance_scale, float):
        # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
        # shape information, as they may be sharded (when `jit` is `True`), or not.
        guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
        if len(prompt_ids.shape) > 2:
            # Assume sharded
            guidance_scale = guidance_scale[:, None]

    print(prompt_ids.shape)
    start_timestep = get_timestep_start(num_inference_steps, strength)
    if jit:
        images = my_p_generate(prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,)
    else:
        images = my_generate( prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,)

    images = np.asarray(images)
    return images

# %%


# %%
rng = jax.random.split(jax.random.PRNGKey(0), jax.device_count())
def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = prepare_inputs(prompt, pipeline.tokenizer)
    neg_prompt_ids = prepare_inputs(neg_prompt, pipeline.tokenizer)
    return prompt_ids, neg_prompt_ids

NUM_DEVICES = jax.device_count()

def replicate_all(prompt_ids, image, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    p_image_ids = replicate(image)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_image_ids, p_neg_prompt_ids, rng

strength=0.75
num_inference_steps=50
height=512
width=768
jit=True

def aot_compile(
    prompt=prompts,
    negative_prompt=neg_prompts,
    image=init_img,
    seed=0,
    guidance_scale=7.5,
    strength=0.75,
    num_inference_steps=50,
    height=512,
    width=768,
):
    prompt_ids, image_ids, neg_prompt_ids = prepare_inputs(prompt, image, negative_prompt, tokenizer=pipeline.tokenizer)
    prompt_ids, image_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, image_ids, neg_prompt_ids, seed)
    start_timestep = get_timestep_start(num_inference_steps, strength)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    # Static argnums are pipe, start_timestep, num_inference_steps, height, width
    # Non-static (0) args are (sharded) input tensors mapped over their first dimension : prompt_ids, image, params, prng_seed,...,guidance_scale, noise, neg_prompt_ids,
    #in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0, 0, 0),
    # static_broadcasted_argnums=(0, 5, 6, 7, 8),)
    #(pipe, prompt_ids, image, params, prng_seed, start_timestep, num_inference_steps, height, width, guidance_scale, noise, neg_prompt_ids,):

    #in_axes=(0, 0, 0, 0, None, None, None, None, 0, 0, 0), 
    # static_broadcasted_argnums=(5, 6, 7, 8),
    return (
        # pmap(pipeline._generate, static_broadcasted_argnums=[4,5, 6, 7]) # (5, 6, 7, 8) are the static args
        pmap(my_generate, static_broadcasted_argnums=[4,5, 6, 7]) # (5, 6, 7, 8) are the static args
        .lower(
                prompt_ids,
                image_ids,
                p_params, 
                rng, 
                start_timestep, # static (4)  
                num_inference_steps,  # static
                height,  # static
                width,  # static
                g, 
                None, # noise
                neg_prompt_ids 
            )
        .compile()
    )

# %%
# Compile
p_generate = aot_compile()



def generate(prompt, image, negative_prompt, seed=0, guidance_scale=7.5):
    prompt_ids, image_ids, neg_prompt_ids = prepare_inputs(prompt, image, negative_prompt, tokenizer=pipeline.tokenizer)
    prompt_ids, image_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, image_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(prompt_ids, image_ids, p_params, rng, g, None, neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return numpy_to_pil(np.array(images))

# %% 
start = time.time()
images = generate(prompts, init_img, neg_prompts)
images[0].show()
print(f"Inference in {time.time() - start}")

# %%
start = time.time()
images = generate(prompts, init_img, neg_prompts)
images[2].show()
print(f"Inference in {time.time() - start}")
# 2.700s 

# %%
# Plot all images

make_image_grid(images, rows=len(images)//4, cols=4)

# %%
