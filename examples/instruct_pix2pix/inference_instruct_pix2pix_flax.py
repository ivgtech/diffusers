# %% 
import os
import sys
import time
import requests
from io import BytesIO

import warnings
from functools import partial
from typing import Any, Tuple, Dict, List, Optional, Union
from packaging import version

import PIL
from PIL import Image
import matplotlib.pyplot as plt
from pickle import UnpicklingError

import jax
import jax.numpy as jnp
import numpy as np 
from jax import pmap

from diffusers.models.unets import unet_spatio_temporal_condition
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.set_cache_dir("/tmp/sd_cache")

NUM_DEVICES = jax.device_count()

from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard, onehot
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict

import torch
from torch.utils.data import DataLoader

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
    FlaxEulerDiscreteScheduler,
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

# Instruct-Pix2Pix
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor


# local imports
# from development.preprocess_load_data_april4 import  train_dataloader_torch, plot_batch
from jax_dataloader import NumpyLoader, train_dataset, show_batch, batch_to_pil_plus_text

DEBUG = False 

# Training data
# NOTE: shapes are (1, 3, 256, 256), (1, 3, 256, 256), (1, 77)
#    dict_keys(['original_pixel_values', 'edited_pixel_values', 'input_ids'])

batch_size = 1   # NOTE: batch size should be 1 for now, as height/width are passed in as ints and not arrays
training_generator = NumpyLoader(
    train_dataset, 
    batch_size=batch_size,
    num_workers= 0
)
batch = next(iter(training_generator))
show_batch(batch)


# Models
dtype = jax.numpy.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    '../flax_models/instruct-pix2pix',
)

vae = pipeline.vae
text_encoder = pipeline.text_encoder
tokenizer = pipeline.tokenizer
unet = pipeline.unet
scheduler = pipeline.scheduler
safety_checker = pipeline.safety_checker
feature_extractor = pipeline.feature_extractor
dtype = pipeline.dtype





# Helper functions

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

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

def prepare_inputs(
    prompt: Union[str, List[str]], 
    image: Union[Image.Image,  List[Image.Image]], 
    negative_prompt: Union[str, List[str], None], 
    tokenizer: CLIPTokenizer
    ):
    if not isinstance(prompt, (str, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
    if not isinstance(negative_prompt, (str, list)) and negative_prompt is not None:
        raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
    if not isinstance(image, (Image.Image, list)):
        raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")
    if isinstance(image, Image.Image):
        image = [image]

    processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])
    text_input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    ).input_ids

    max_length = text_input_ids.shape[-1]

    if negative_prompt is None:
        uncond_input_ids = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
        ).input_ids
    else:
        neg_text_input_ids = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        ).input_ids
        uncond_input_ids = neg_text_input_ids

    return text_input_ids, processed_images, uncond_input_ids

def numpy_to_pil(images):
    #  from src/diffusers/pipelines/pipeline_flax_utils.py
    if images.ndim == 3: images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1: # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image  

    

def _encode_prompt(
    prompt,
    num_images_per_prompt,
    negative_prompt,
):
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    # Encode prompt
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="jax")
    prompt_embeds = text_encoder(text_inputs.input_ids).last_hidden_state

    # Handle negative prompt
    if negative_prompt is not None:
        negative_text_inputs = tokenizer(negative_prompt, padding="max_length",
                                                max_length=tokenizer.model_max_length,
                                                truncation=True, return_tensors="jax")
        negative_prompt_embeds = text_encoder(negative_text_inputs.input_ids).last_hidden_state
    else:
        raise ValueError("negative_prompt is required when do_classifier_free_guidance is True")

    # Reshape prompt embeddings for generation
    prompt_embeds = jnp.tile(prompt_embeds, (1, num_images_per_prompt, 1))
    prompt_embeds = prompt_embeds.reshape((batch_size * num_images_per_prompt,) + prompt_embeds.shape[1:])

    # Reshape negative prompt embeddings for guidance
    negative_prompt_embeds = jnp.tile(negative_prompt_embeds, (1, num_images_per_prompt, 1))
    negative_prompt_embeds = negative_prompt_embeds.reshape((batch_size * num_images_per_prompt,) + negative_prompt_embeds.shape[1:])

    # Concatenate embeddings for classifier-free guidance
    prompt_embeds = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

    return prompt_embeds


def encode_prompt(
    prompt_batch: List[str],
    text_encoders: List[FlaxCLIPTextModel],
    tokenizers: List[CLIPTokenizer],
    proportion_empty_prompts: float,
    rng: jax.random.PRNGKey,
    is_train: bool = True
) -> (jnp.ndarray, jnp.ndarray):
    prompt_embeds_list = []
    captions = []
    rng, sub_rng = jax.random.split(rng)

    for caption in prompt_batch:
        if jax.random.uniform(sub_rng) < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # split rng again for random choices
            rng, choice_rng = jax.random.split(rng)
            chosen_index = jax.random.randint(choice_rng, (1,), 0, len(caption))
            captions.append(caption[chosen_index[0]] if is_train else caption[0])

    pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="jax"
        )
        text_input_ids = text_inputs.input_ids
        # pass the text_encoder params to the text_encoder explicitly
        outputs = text_encoder(text_input_ids, params=params["text_encoder"])
        prompt_embeds = outputs.last_hidden_state

        # handling pooled outputs is model-specific, here we assume last_hidden_state for simplicity
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = prompt_embeds[:, 0, :]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = jnp.concatenate(prompt_embeds_list, axis=-1)
    # For pooled prompt embeddings: here we assume the first token's embedding is used as the pooled representation
    pooled_prompt_embeds = prompt_embeds[:, 0, :].reshape(prompt_embeds.shape[0], -1)
    
    return prompt_embeds, pooled_prompt_embeds






# %%
# (2) Model functions



# Instruct Pix2Pix inference function

def my_generate(
    prompt_ids: jnp.ndarray,
    image: jnp.ndarray,
    params: Union[Dict, FrozenDict],
    prng_seed: jax.Array,
    num_inference_steps: int = 100,
    height: int = None,
    width: int = None,
    guidance_scale: Union[float, jnp.ndarray] = 7.5,
    image_guidance_scale: Union[float, jnp.ndarray] = 1.5,
    latents: Optional[jnp.ndarray] = None,
    neg_prompt_ids: Optional[jnp.ndarray] = None,
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    # get prompt text embeddings
    prompt_embeds = text_encoder(prompt_ids, params=params["text_encoder"])[0]

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
    
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    # pix2pix has two negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
    context = jnp.concatenate([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds ])
    #context = _encode_prompt(prompt_ids, 1, neg_prompt_ids) 

    # Ensure model output will be `float32` before going into the scheduler
    # guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

    num_channels_latents = vae.config.latent_channels
    latents_shape = (
        batch_size,
        num_channels_latents,
        # unet.config.in_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    if latents is None:
        latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
    else:
        if latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

    # Create image latents
    image_latents = prepare_image_latents(image, params, batch_size, 1, True, prng_seed)
    image_latents = image_latents.transpose((0, 3, 1, 2))
    # Create init_latents
    init_latent_dist = vae.apply({"params": params["vae"]}, image, method=vae.encode).latent_dist
    init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))
    init_latents = vae.config.scaling_factor * init_latents

    def loop_body(step, args):
        latents, scheduler_state = args
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes

        # Expand the latents if we are doing classifier free guidance.
        # The latents are expanded 3 times because for pix2pix the guidance\
        # is applied for both the text and the input image.
        latents_input = jnp.concatenate([latents] * 3)

        ###
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]

        # concat latents, image_latents in the channel dimension
        timestep = jnp.broadcast_to(t, latents_input.shape[0])
        latents_input = scheduler.scale_model_input(scheduler_state, latents_input, t)

        latents_input = jnp.concatenate([latents_input, image_latents], axis=1)
        ###

        # t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
        # timestep = jnp.broadcast_to(t, latents_input.shape[0])

        # latents_input = scheduler.scale_model_input(scheduler_state, latents_input, t)


        # predict the noise residual
        noise_pred = unet.apply(
            {"params": params["unet"]},
            jnp.array(latents_input),
            jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=context,
        ).sample

        noise_pred_text, noise_pred_image, noise_pred_uncond = jnp.split(noise_pred, 3, axis=0)

        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_image)
            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )
        # perform guidance
        # noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
        # noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
        return latents, scheduler_state

    # set the scheduler state
    scheduler_state = scheduler.set_timesteps(
        params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
    )

    # latent_timestep = scheduler_state.timesteps[start_timestep : start_timestep + 1].repeat(batch_size)
    # # timestep index for the latents (convert from float32 to int32)
    # latent_timestep = jnp.array(latent_timestep, dtype=jnp.int32)
    # latents = scheduler.add_noise(params["scheduler"], init_latents, latents, latent_timestep) # TODO: exception here


    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * params["scheduler"].init_noise_sigma

    if DEBUG:
        # run with python for loop
        for i in range(num_inference_steps):
            latents, scheduler_state = loop_body(i, (latents, scheduler_state))
    else:
        latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

    # scale and decode the image latents with vae
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.apply({"params": params["vae"]}, latents, method=vae.decode).sample
    image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
    return image



# Replicate mutable arrays across devices before passing them to the AOT compiled function
def replicate_all(prompt_ids, image, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    p_image_ids = replicate(image)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_image_ids, p_neg_prompt_ids, rng

def get_timestep_start( num_inference_steps, strength): 
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    return max(num_inference_steps - init_timestep, 0)


    
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents_jax(image, params, key=None, sample_mode="sample"):
    # pass the image through the VAE to get the latents
    encoder_output = vae.apply({"params": params["vae"]}, image, method=vae.encode)

    # get the latent distribution
    latent_dist = encoder_output.latent_dist 

    if sample_mode == "sample":
        return latent_dist.sample(key)
    elif sample_mode == "argmax":
        return latent_dist.mode()
    elif sample_mode == "latents":
        return encoder_output.latents
    else:
        raise ValueError(f"Invalid sample_mode: {sample_mode}")

# convert helper functions to jax.numpy
def prepare_image_latents(image, params, batch_size, num_images_per_prompt, do_classifier_free_guidance, key):

    if image.shape[1] == 4:
        image_latents = image
    else:
        image_latents = retrieve_latents_jax(image, params, key, sample_mode="argmax")

    batch_size_adjusted = batch_size * num_images_per_prompt
    # Handling different batch sizes
    if batch_size_adjusted > image_latents.shape[0] and batch_size_adjusted % image_latents.shape[0] == 0:
        additional_image_per_prompt = batch_size_adjusted // image_latents.shape[0]
        image_latents = jnp.concatenate([image_latents] * additional_image_per_prompt, axis=0)
    elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
        raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
    else:
        image_latents = jnp.concatenate([image_latents], axis=0)

    if do_classifier_free_guidance:
        uncond_image_latents = jnp.zeros_like(image_latents)
        # NOTE: Instruct Pix2Pix conditions on both text, image and condition free guidance
        image_latents = jnp.concatenate([image_latents, image_latents, uncond_image_latents], axis=0)


    print("return image_latents.shape", image_latents.shape)
    return image_latents













# %%

# (3) Ahead of time compilation 

# original_images, edited_images, prompt_ids = batch['original_pixel_values'], batch['edited_pixel_values'], batch['input_ids']
# pil_images, _ , text_prompts = batch_to_pil_plus_text(batch, tokenizer)

# raw_neg_prompts = [""] * len(prompt_ids)
# neg_prompts = tokenizer(raw_neg_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np").input_ids

# height, width = pil_images[0].size  # NOTE: height and width are the same for all images in the batch



def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")
url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
image = download_image(url).resize((512, 512))
prompt = "make the mountains snowy"


height, width = image.size
pil_images = image
text_prompts = prompt

image_ids = preprocess(pil_images, jnp.float32)
prompt_ids = tokenizer(text_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np").input_ids
neg_prompt_ids  = None

num_inference_steps = 100 
image_guidance_scale = 1.2
guidance_scale = 7.5
strength = 1.0
seed = 1371 
start_timestep = get_timestep_start(num_inference_steps, strength)
p_params = replicate(params)


prompt_ids, image_ids, neg_prompt_ids = prepare_inputs(prompt, image, negative_prompt=None, tokenizer=pipeline.tokenizer)

def aot_compile(
    prompt_ids=prompt_ids,
    image_ids=image_ids,
    seed=seed,
):
    p_prompt_ids = replicate(prompt_ids)
    p_image_ids = replicate(image_ids)
    rng = jax.random.PRNGKey(seed)
    p_rng = jax.random.split(rng, NUM_DEVICES)

    return (
        pmap(my_generate, static_broadcasted_argnums=[4, 5, 6, 7, 8]) 
        .lower(
                p_prompt_ids,
                p_image_ids,
                p_params,
                p_rng,
                num_inference_steps, # static (4)
                height, # static (5)
                width, # static (6)
                guidance_scale, # static (7)
                image_guidance_scale, # static (8)
        )
        .compile()
    )



# %%

# Initial compilation 
start = time.time()
p_generate = aot_compile()
print(f"Inference in {time.time() - start}")
# 102.0s if not cc cached, otherwise >12s

# Here we use a curried function to avoid recompiling the calling function every time we want to generate an image
negative_prompt = None
def generate(prompt, image, seed=0):
    prompt_ids, image_ids, _ = prepare_inputs(prompt, image, negative_prompt=None, tokenizer=pipeline.tokenizer)
    p_prompt_ids = replicate(prompt_ids)
    p_image_ids = replicate(image_ids)

    rng = jax.random.PRNGKey(seed)
    p_rng = jax.random.split(rng, NUM_DEVICES)

    images = p_generate(p_prompt_ids, p_image_ids, p_params, p_rng)
    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return numpy_to_pil(np.array(images))


# Middle of the road inference time
start = time.time()
images = generate(text_prompts, pil_images)
print(f"Inference in {time.time() - start}")
# ~3.05

# Fastest as function is fully cached
start = time.time()
images = generate(text_prompts, pil_images)
print(f"Inference in {time.time() - start}")
# ~2.71s 

# Plot all images
image.show()
make_image_grid(images, rows=len(images)//4, cols=4)

# %%

# (4) Editing a different image and prompt (no recompilation) 


img_path = '/home/v/instruct-pix2pix/imgs/example.jpg'
with open(img_path, 'rb') as f:
    image = Image.open(f).convert("RGB").resize((512, 512))
prompt = "turn him into cyborg"

start = time.time()
images = generate(prompt, image)
print(f"Inference in {time.time() - start}")
# ~2.7s 

image.show()
make_image_grid(images, rows=len(images)//4, cols=4)

# %%

pil_images.save('images/example_mountains.png')
title = 'mountains'
for i, img in enumerate(images):  
    img.save(f"images/april_11_{title}{i}.png")
# %%