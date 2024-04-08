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

import PIL
from PIL import Image
import matplotlib.pyplot as plt
from pickle import UnpicklingError
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np 
from jax import pmap
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.set_cache_dir("/tmp/sd_cache")

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

# Instruct-Pix2Pix
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor


# local imports
from flax_from_hf_pretrained_april4 import get_pretrained
from model_converter import load_from_standard_weights
from preprocess_load_data_april4 import train_dataset, train_dataloader, plot_batch

DEBUG = True 


# %%
# Let's cache the model compilation, so that it doesn't take as long the next time around.
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.set_cache_dir("/tmp/sd_cache")

NUM_DEVICES = jax.device_count()

dtype = jnp.bfloat16
pipeline, params = FlaxDiffusionPipeline.from_pretrained(
    '../flax_models/instruct-pix2pix',
    revision="flax",
    dtype=dtype,
)




# %%

# Preprocessing and helper functions 

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

    
# %% 

#####################################################################################
# train_controlnet_sdxl.py

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, rng, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:

        #if jax.random.uniform(rng) < proportion_empty_prompts:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# %%




import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import FlaxCLIPTextModel, CLIPTokenizer

import jax
import jax.numpy as jnp
from typing import List, Optional
from transformers import CLIPTokenizer, FlaxCLIPTextModel

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

        outputs = text_encoder(text_input_ids)
        prompt_embeds = outputs.last_hidden_state

        # handling pooled outputs is model-specific, here we assume last_hidden_state for simplicity
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = prompt_embeds[:, 0, :]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = jnp.concatenate(prompt_embeds_list, axis=-1)
    # For pooled prompt embeddings: here we assume the first token's embedding is used as the pooled representation
    pooled_prompt_embeds = prompt_embeds[:, 0, :].reshape(prompt_embeds.shape[0], -1)
    
    return prompt_embeds, pooled_prompt_embeds


prompt_batch = ["A photo of a dog in the park", "A painting of a sunset"] 
# Generate a random key
rng = jax.random.PRNGKey(0) 
# Encode the prompts
prompt_embeds, pooled_prompt_embeds = encode_prompt(
    prompt_batch, [text_encoder], [tokenizer], 0.5, rng, is_train=True
)

print(prompt_embeds.shape)
print(pooled_prompt_embeds.shape)

# %% 
#plot_batch(train_dataloader, pipeline.tokenizer)
batch = next(iter(train_dataloader))
prompt_batch = batch["input_ids"]
proportion_empty_prompts = 0.5
is_train = True
#encode_prompt(prompt_batch, [pipeline.text_encoder], [pipeline.tokenizer], proportion_empty_prompts, is_train=True)
prompt_embeds, pooled_prompt_embeds = encode_prompt(
    prompt_batch, [text_encoder], [tokenizer], proportion_empty_prompts, rng, is_train=True 
    )
print(prompt_embeds.shape)
print(pooled_prompt_embeds.shape)

# %%


#####################################################################################
# pipeline_stable_diffusion_instruct_pix2pix.py

def _encode_prompt(
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
):
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        # textual inversion: process multi-vector tokens if necessary
        if isinstance( TextualInversionLoaderMixin):
            prompt = maybe_convert_prompt(prompt, tokenizer)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0]

    if text_encoder is not None:
        prompt_embeds_dtype = text_encoder.dtype
    else:
        prompt_embeds_dtype = unet.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        # textual inversion: process multi-vector tokens if necessary
        if isinstance( TextualInversionLoaderMixin):
            uncond_tokens = maybe_convert_prompt(uncond_tokens, tokenizer)

        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # pix2pix has two negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
        prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])

    return prompt_embeds





#####################################################################################
# /home/v/diffusers/examples/community/llm_grounded_diffusion.py


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
def _encode_prompt(
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    lora_scale: Optional[float] = None,
    **kwargs,
):
    deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
    deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

    prompt_embeds_tuple = encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        **kwargs,
    )

    # concatenate for backwards comp
    prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

    return prompt_embeds






# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
def encode_image( image, device, num_images_per_prompt, output_hidden_states=None):
    dtype = next(image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_enc_hidden_states = image_encoder(
            torch.zeros_like(image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
def run_safety_checker( image, device, dtype):
    if safety_checker is None:
        has_nsfw_concept = None
    else:
        if torch.is_tensor(image):
            feature_extractor_input = image_processor.postprocess(image, output_type="pil")
        else:
            feature_extractor_input = image_processor.numpy_to_pil(image)
        safety_checker_input = feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
        image, has_nsfw_concept = safety_checker(
            images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
        )
    return image, has_nsfw_concept

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs( generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
def decode_latents( latents):
    deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
    deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def check_inputs(
    prompt,
    callback_steps,
    negative_prompt=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    callback_on_step_end_tensor_inputs=None,
):
    if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
            f" {type(callback_steps)}."
        )

    if callback_on_step_end_tensor_inputs is not None and not all(
        k in _callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
    ):
        raise ValueError(
            f"`callback_on_step_end_tensor_inputs` has to be in {_callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in _callback_tensor_inputs]}"
        )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
            f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
        )

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                f" {negative_prompt_embeds.shape}."
            )

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
def prepare_latents( batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    return latents

def prepare_image_latents(
     image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        image_latents = image
    else:
        image_latents = retrieve_latents(vae.encode(image), sample_mode="argmax")

    if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
        # expand image_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
        additional_image_per_prompt = batch_size // image_latents.shape[0]
        image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        image_latents = torch.cat([image_latents], dim=0)

    if do_classifier_free_guidance:
        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

    return image_latents

def guidance_scale():
    return _guidance_scale

def image_guidance_scale():
    return _image_guidance_scale

def num_timesteps():
    return _num_timesteps

# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
# corresponds to doing no classifier free guidance.
def do_classifier_free_guidance():
    return guidance_scale > 1.0 and image_guidance_scale >= 1.0









# %% 
# Image and prompt inputs
url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"

image = download_image(url)
prompt = "wipe out the lake"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

# edited_image = pipe(prompt, 
#     image=image, 
#     num_inference_steps=num_inference_steps, 
#     image_guidance_scale=image_guidance_scale, 
#     guidance_scale=guidance_scale,
#     generator=generator,
# ).images[0]

# edited_image.save("edited_image.png")


# %% 
# Inference function run as part of the AOT compilation

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
    latents = scheduler.add_noise(params["scheduler"], init_latents, noise, latent_timestep) # TODO: exception here

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
# Ahead of time compilation (AOT)

# Replicate mutable arrays across devices before passing them to the AOT compiled function
def replicate_all(prompt_ids, image, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    p_image_ids = replicate(image)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_image_ids, p_neg_prompt_ids, rng

# Replicate all model state across devices before passing them to the AOT compiled function
p_params = replicate(params)

def get_timestep_start( num_inference_steps, strength): # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    return max(num_inference_steps - init_timestep, 0)

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
        # NOTE: generate is a functional version of pipeline._generate
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
# Initial compilation 

start = time.time()
p_generate = aot_compile()
print(f"Inference in {time.time() - start}")
# 102.0s if not cc cached, otherwise >12s

# %%

# Here we use a curried function to avoid recompiling the calling function every time we want to generate an image
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
# Middle of the road inference time
start = time.time()
images = generate(prompts, init_img, neg_prompts)
images[0].show()
print(f"Inference in {time.time() - start}")
# ~3.05

# %%
# Compiled function is now fully cached and so inference is at optimal speeds
start = time.time()
images = generate(prompts, init_img, neg_prompts)
images[2].show()
print(f"Inference in {time.time() - start}")
# ~2.71s 

# %%
# Plot all images
make_image_grid(images, rows=len(images)//4, cols=4)

# %%
