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

from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from diffusers.models.modeling_utils import load_state_dict





from diffusers.utils import deprecate, logging, replace_example_docstring
from diffusers.pipelines import FlaxDiffusionPipeline
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker_flax import FlaxStableDiffusionSafetyChecker

# local imports
from flax_from_hf_pretrained_april4 import get_pretrained
from model_converter import load_from_standard_weights


#%%

###############################################################################
#                                                    Save flax pipeline to disk
###############################################################################

dtype = jnp.bfloat16
MODEL_NAME="runwayml/stable-diffusion-v1-5"

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="flax",
    dtype=dtype,
)

safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker", from_pt=True)

pipeline = FlaxStableDiffusionPipeline(
    text_encoder=pipeline.text_encoder,
    vae=pipeline.vae,
    unet=pipeline.unet,
    tokenizer=pipeline.tokenizer, # FlaxPreTrainedModel.save_pretrained of PreTrainedTokenizer
    scheduler=pipeline.scheduler, # FlaxSchedulerMixin.save_pretrained of FlaxEulerDiscreteScheduler 
    safety_checker=safety_checker,
    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
)

pipeline.save_pretrained(
    save_directory= '../flax_models/stable-diffusion-v1-5',
    params={
        "text_encoder": params["text_encoder"],  # FlaxPreTrainedModel.save_pretrained
        "vae": params["vae"],
        "unet": params["unet"],
        "safety_checker": safety_checker.params,
    },
)





# %%

###############################################################################
#                                    Flax inference from locally saved pipeline
###############################################################################

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















class FlaxSD(FlaxDiffusionPipeline):
    DEBUG = False

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

    def prepare_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _generate(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        latents: Optional[jnp.ndarray] = None,
        neg_prompt_ids: Optional[jnp.ndarray] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

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

        # Ensure model output will be `float32` before going into the scheduler
        guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

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

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents.shape
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma

        if DEBUG:
            # run with python for loop
            for i in range(num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(0, num_inference_steps, loop_body, (latents, scheduler_state))

        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image

    def __call__(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Union[float, jnp.ndarray] = 7.5,
        latents: jnp.ndarray = None,
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

        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
            )
        else:
            images = self._generate(
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
            )

           
        images = np.asarray(images)
        has_nsfw_concept = False
        if not return_dict:
            return (images, has_nsfw_concept)

        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


# Static argnums are pipe, num_inference_steps, height, width. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, None, None, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 4, 5, 6),
)
def _p_generate(
    pipe,
    prompt_ids,
    params,
    prng_seed,
    num_inference_steps,
    height,
    width,
    guidance_scale,
    latents,
    neg_prompt_ids,
):
    return pipe._generate(
        prompt_ids,
        params,
        prng_seed,
        num_inference_steps,
        height,
        width,
        guidance_scale,
        latents,
        neg_prompt_ids,
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))
def _p_get_has_nsfw_concepts(pipe, features, params):
    return pipe._get_has_nsfw_concepts(features, params)


def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)

# %%


sd = FlaxSD(
    vae=pipeline.vae,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer,
    unet=pipeline.unet,
    scheduler=pipeline.scheduler,
    safety_checker=pipeline.safety_checker,
    feature_extractor=pipeline.feature_extractor,
    dtype=dtype,
)

# %%


dtype = jnp.bfloat16
pipeline, params = FlaxSD.from_pretrained(
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

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

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
