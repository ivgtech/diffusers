# %% 
import os
import math
import PIL
import copy
import argparse
import logging
import requests
import random
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.training.common_utils import shard
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

import torch
import torch.utils.checkpoint
from torchvision import transforms
import transformers
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version

from jax_dataloader import NumpyLoader, train_dataset 


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = logging.getLogger(__name__)

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

# convert the namespace to a dictionary
args = {
'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
'revision': 'bf16',
'variant': None,
'dataset_name': 'fusing/instructpix2pix-1000-samples',
'dataset_config_name': None,
'train_data_dir': None,
'original_image_column': 'input_image',
'edited_image_column': 'edited_image',
'edit_prompt_column': 'edit_prompt',
'val_image_url': None,
'validation_prompt': None,
'num_validation_images': 4,
'validation_epochs': 1,
'max_train_samples': None,
'output_dir': 'instruct-pix2pix-model',
'cache_dir': None,
'seed': 42,
'resolution': 4,
'center_crop': False,
'random_flip': True,
'train_batch_size': 4,
'num_train_epochs': 100,
'max_train_steps': 15000,
'gradient_accumulation_steps': 4,
'gradient_checkpointing': True,
'learning_rate': 5e-05,
'scale_lr': False,
'lr_scheduler': 'constant',
'lr_warmup_steps': 0,
'conditioning_dropout_prob': 0.05,
'use_8bit_adam': False,
'allow_tf32': False,
'use_ema': False,
'non_ema_revision': None,
'dataloader_num_workers': 0,
'adam_beta1': 0.9,
'adam_beta2': 0.999,
'adam_weight_decay': 0.01,
'adam_epsilon': 1e-08,
'max_grad_norm': 1.0,
'push_to_hub': True,
'hub_token': None,
'hub_model_id': None,
'logging_dir': 'logs',
'mixed_precision': 'bf16',
'report_to': 'tensorboard',
'local_rank': -1,
'checkpointing_steps': 5000,
'checkpoints_total_limit': 1,
'resume_from_checkpoint': None,
'enable_xformers_memory_efficient_attention': True,
'from_pt': False,
'max_ema_decay': 0.999,
'min_ema_decay': 0.5,
'ema_decay_power': 0.6666666,
'ema_inv_gamma': 1.0,
'start_ema_update_after': 100,
'update_ema_every': 10,
}

class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries) 
args = Args(**args)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,)
# Setup logging, we only want one process per machine to log things on the screen.
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
if jax.process_index() == 0:
    transformers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()

if args.seed is not None: set_seed(args.seed)


# (1) Data loader: Creates a JAX generator from a standard PyTorch dataloader
train_dataloader = NumpyLoader(
    train_dataset, 
    batch_size=args.train_batch_size,
    num_workers= 0
)


# (5) Conditioning dropout implementation

# %% 
def xapply_conditioning_dropout(encoder_hidden_states, original_image_embeds, dropout_rng, bsz, args):
    print("Encoder hidden states: ", encoder_hidden_states.shape, ' original_image_embeds.shape, ', original_image_embeds.shape, 'BSZ: ', bsz)

    # Make sure bsz is a static, concrete number
    random_p = jax.random.uniform(dropout_rng, shape=(bsz,), minval=0.0, maxval=1.0)

    print("Random p: ", random_p, random_p.shape)

    prompt_mask = random_p < 2 * args.conditioning_dropout_prob

    print("Prompt mask: ", prompt_mask, prompt_mask.shape)
    print('Tokenize captions: ', tokenize_captions([""]).shape)

    null_text_conditioning = text_encoder(tokenize_captions([""]), params=text_encoder.params, train=False)[0]

    print("Null text conditioning: ", null_text_conditioning.shape)
    print(prompt_mask[:, None, None].shape, null_text_conditioning.shape, encoder_hidden_states.shape   )

    encoder_hidden_states = jnp.where(prompt_mask[:, None, None], null_text_conditioning, encoder_hidden_states)

    print("Encoder hidden states: ", encoder_hidden_states.shape)

    image_mask = (random_p >= args.conditioning_dropout_prob) & (random_p < 3 * args.conditioning_dropout_prob)

    print("Image mask: ", image_mask, image_mask.shape)

    image_mask = image_mask[:, None, None, None]  # Adjust the mask shape to match the embeddings shape

    print("Image mask: ", image_mask, image_mask.shape) 

    original_image_embeds = original_image_embeds * image_mask
    print("Original image embeds: ",original_image_embeds.shape)

    return encoder_hidden_states, original_image_embeds

def tokenize_captions(captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="np")
    return inputs.input_ids

def apply_conditioning_dropout(encoder_hidden_states, original_image_embeds, dropout_rng, bsz, args):

    # Make sure bsz is a static, concrete number
    # Generating a random tensor `random_p` with shape (bsz,)
    random_p = jax.random.uniform(dropout_rng , (bsz,))

    # Generating the prompt mask
    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
    prompt_mask = prompt_mask.reshape(bsz, 1, 1)  # Reshape to match dimensions for broadcasting
    null_text_conditioning = text_encoder(tokenize_captions([""]), params=text_encoder.params, train=False)[0]

    # Applying null conditioning using the prompt mask
    updated_encoder_hidden_states = jnp.where(prompt_mask, null_text_conditioning, encoder_hidden_states)

    # Generating the image mask
    image_mask_dtype = original_image_embeds.dtype
    image_mask = 1 - (
        (random_p >= args.conditioning_dropout_prob).astype(image_mask_dtype)
        * (random_p < 3 * args.conditioning_dropout_prob).astype(image_mask_dtype)
    )
    image_mask = image_mask.reshape(bsz, 1, 1, 1)
    
    # Final image conditioning.
    original_image_embeds = image_mask * original_image_embeds
    
    return updated_encoder_hidden_states, original_image_embeds

# apply_conditioning_dropout(jnp.ones((4,77,768)), jnp.ones((4,4,32,32)), jax.random.PRNGKey(42), 4, args);







# (2) Models and state

weight_dtype = jnp.float32
if args.mixed_precision == "fp16":
    weight_dtype = jnp.float16
elif args.mixed_precision == "bf16":
    weight_dtype = jnp.bfloat16


# NOTE: Hugging Face uses variants (`bfloat16` or `flax`) to refer to different repository branches and hence different checkpoints. 
# The `bfloat16` branch/variant refers to `EMA only weights` (parameters where smoothed out during training) and `flax` refers to the `raw weights` (no EMA smoothing used during training).

# That is, the difference between the two checkpoints is the whether the EMA smoothing was applied or not:
# v1-5-pruned-emaonly.ckpt: This checkpoint uses only the EMA weights. EMA is a technique used during training to stabilize and improve the convergence properties of the model. It smooths the model parameters over the course of training by maintaining a running average of model weights. Using EMA weights typically results in a model that generalizes better on new, unseen data.
# v1-5-pruned.ckpt: This checkpoint likely includes the "raw" final weights obtained at the end of training, without applying the EMA smoothing. This means it could potentially be less smooth or stable compared to the EMA-only model, but it might capture nuances that the EMA-smoothed model does not.

# See timothybrooks/instruct-pix2pix/scripts/download_pretrained_sd.sh for the exact checkpoints used:
# https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
# https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt

assert args.revision=="bf16", "Instruct-Pix2Pix was fine-tuned on a v1-5-pruned-emaonly.ckpt checkpoint, which corresonds to Hugging Face's git identifier revision='bf16'. Please set the revision to 'bf16'."
# NOTE: OSError: bfloat16 is not a valid git identifier (branch name, tag name or commit id) that exists for this model name. Check the model page at 'https://huggingface.co/runwayml/stable-diffusion-v1-5' for available revisions.
assert weight_dtype==jnp.bfloat16, "Please set the mixed_precision to 'bf16' to match the model's weights."

tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    from_pt=args.from_pt,
    revision="bf16", # corresponds to bfloat16 precision, use flax for float32
    subfolder="tokenizer",
)
text_encoder = FlaxCLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path,
    from_pt=args.from_pt,
    revision="bf16", # corresponds to bfloat16 precision, use flax for float32
    subfolder="text_encoder",
    dtype=weight_dtype,
)
vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path,
    from_pt=args.from_pt,
    revision="flax", # NOTE: the vae model uses the `flax` revision for the raw weights (no EMA smoothing)
    subfolder="vae",
    dtype=weight_dtype,
)

# Load the converted unet model
save_dir = 'modified_unet'
unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    save_dir,
    dtype=jnp.bfloat16 # NOTE: from a EMA-only checkpoint, in fp32 precision, converted to Flax using the save_unet.py script, and then loaded in bfloat16 precision here!
)


# (3) Helper functions

def get_nparams(params: FrozenDict) -> int:
    nparams = 0
    for item in params:
        if isinstance(params[item], FrozenDict) or isinstance(params[item], dict):
            nparams += get_nparams(params[item])
        else:
            nparams += params[item].size
    return nparams

def retrieve_latents_jax(image, vae, vae_params, key=None, sample_mode="sample"):
    # pass the image through the VAE to get the latents
    encoder_output = vae.apply(
                {"params": vae_params}, image, deterministic=True, method=vae.encode)

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


# (4) EMA implementation

# def get_decay(
#     step: int, max_ema_decay: float = 0.9999,
#     min_ema_decay: float = 0.0,
#     ema_inv_gamma: float = 1.0,
#     ema_decay_power: float = 2 / 3, 
#     use_ema_warmup: bool = False,
#     start_ema_update_after_n_steps: float = 10.0
#     ):
#     # Adjust step to consider the start update offset
#     adjusted_step = jnp.maximum(step - start_ema_update_after_n_steps, 0)

#     # Compute base decay
#     if use_ema_warmup:
#         decay = 1.0 - (1.0 + adjusted_step / ema_inv_gamma) ** -ema_decay_power
#     else:
#         initial_steps = jnp.where(start_ema_update_after_n_steps == 0, 10.0, start_ema_update_after_n_steps)
#         decay = (1.0 + adjusted_step) / (initial_steps + adjusted_step)

#     # Ensure decay starts changing only after certain steps
#     decay = jnp.where(step > start_ema_update_after_n_steps, decay, min_ema_decay)

#     # Clip the decay to ensure it stays within the specified bounds
#     return jnp.clip(decay, min_ema_decay, max_ema_decay)

# class EMAState(struct.PyTreeNode):
#     ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
#     # add a decay function to the EMAState
#     decay_fn: Callable[[float], "EMAState"] = struct.field(pytree_node=False)

# class ExtendedTrainState(TrainState, EMAState):
#     def apply_gradients(self, grads: core.FrozenDict[str, Any], ema_decay: float = 0.999) -> "ExtendedTrainState":
#         # Standard parameter update
#         updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
#         new_params = optax.apply_updates(self.params, updates)
        
#         ema_decay = self.decay_fn( self.step,) 

#         # Update EMA parameters
#         new_ema_params = jax.tree.map(
#             lambda ema, p: ema * ema_decay + (1 - ema_decay) * p, self.ema_params, new_params
#         )
        
#         return self.replace(params=new_params, opt_state=new_opt_state, ema_params=new_ema_params)




#def main():


# Optimization

total_train_batch_size = args.train_batch_size * jax.local_device_count()
if args.scale_lr:
    args.learning_rate = args.learning_rate * total_train_batch_size

constant_scheduler = optax.constant_schedule(args.learning_rate)

adamw = optax.adamw(
    learning_rate=constant_scheduler,
    b1=args.adam_beta1,
    b2=args.adam_beta2,
    eps=args.adam_epsilon,
    weight_decay=args.adam_weight_decay,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    adamw,
)

print("Number of unet parameters: ", get_nparams(unet_params))

# # Initialize EMA params with original model params
# ema_params = copy.deepcopy(unet_params)
# print("Number of EMA parameters: ", get_nparams(ema_params))

# Prepare optimizer and state, including EMA parameters
# state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
# state = ExtendedTrainState.create(apply_fn=unet.__call__, params=unet_params, ema_params=ema_params, decay_fn=get_decay, tx=optimizer)

# Simplified decay function (matching PyTorch's EMAModel)
def get_decay(step: int, decay: float = 0.9999):
    return decay

# Extended decay function 
def get_decay(
    step: int, max_ema_decay: float = 0.9999,
    min_ema_decay: float = 0.0,
    ema_inv_gamma: float = 1.0,
    ema_decay_power: float = 2 / 3, 
    use_ema_warmup: bool = False,
    start_ema_update_after_n_steps: float = 10.0
    ):
    """
    Computes the EMA decay factor based on the current step and a dynamic decay rate.
    """
    # Adjust step to consider the start update offset
    adjusted_step = jnp.maximum(step - start_ema_update_after_n_steps, 0)

    # Compute base decay
    if use_ema_warmup:
        decay = 1.0 - (1.0 + adjusted_step / ema_inv_gamma) ** -ema_decay_power
    else:
        initial_steps = jnp.where(start_ema_update_after_n_steps == 0, 10.0, start_ema_update_after_n_steps)
        decay = (1.0 + adjusted_step) / (initial_steps + adjusted_step)

    # Ensure decay starts changing only after certain steps
    decay = jnp.where(step > start_ema_update_after_n_steps, decay, min_ema_decay)

    # Clip the decay to ensure it stays within the specified bounds
    return jnp.clip(decay, min_ema_decay, max_ema_decay)
#     return jnp.clip(decay, min_ema_decay, max_ema_decay)

@jax.jit
def ema_update(new_params, ema_params, ema_decay):
    # Update EMA parameters
    # return jax.tree.map(lambda ema, p: ema * decay + p * (1 - decay), ema_params, params)
    new_ema_params = jax.tree.map(
        lambda ema, p: ema * ema_decay + (1 - ema_decay) * p, ema_params, new_params
    )
    return new_ema_params

class ExtendedTrainState(TrainState):
    ema_params: core.FrozenDict[str, Any]
    #decay_function: Callable[[int], float]

    def apply_gradients(self, grads: core.FrozenDict[str, Any]) -> "ExtendedTrainState":
        # Standard parameter update
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        # Update EMA parameters using the decay function
        # decay = self.decay_fn(self.step)  # Using extended_get_decay
        #decay = get_decay(self.step)  # NOTE: set decay function here
        # new_ema_params = ema_update(new_params, self.ema_params, decay)

        return self.replace(params=new_params, opt_state=new_opt_state, ema_params=self.ema_params)


# Initialize EMA params with original model params
ema_params = copy.deepcopy(unet_params)

# Prepare optimizer and state
state = ExtendedTrainState.create(
    apply_fn=unet.__call__, params=unet_params, ema_params=ema_params, tx=optimizer
)

noise_scheduler = FlaxDDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
noise_scheduler_state = noise_scheduler.create_state()

# Initialize our training
rng = jax.random.PRNGKey(args.seed)
train_rngs = jax.random.split(rng, jax.local_device_count())

# Training function 
def train_step(state, text_encoder_params, vae_params, batch, train_rng):
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

    def compute_loss(params):
        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": vae_params}, batch["edited_pixel_values"], deterministic=True, method=vae.encode
        )
        latents = vae_outputs.latent_dist.sample(sample_rng)
        latents = jnp.einsum("ijkl->iljk", latents) * vae.config.scaling_factor  # (NHWC) -> (NCHW)
        noise_rng, timestep_rng = jax.random.split(sample_rng)

        # Sample noise that we'll add to the latents
        noise = jax.random.normal(noise_rng, latents.shape)

        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = jax.random.randint(
            timestep_rng,
            (bsz,),
            0,
            noise_scheduler.config.num_train_timesteps,
        )
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(
            batch["input_ids"],
            params=text_encoder_params,
            train=False,
        )[0]

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        vae_image_outputs = vae.apply(
            {"params": vae_params}, batch["original_pixel_values"], deterministic=True, method=vae.encode
        )
        original_image_embeds = vae_image_outputs.latent_dist.mode()
        original_image_embeds = jnp.einsum("ijkl->iljk", original_image_embeds) # (NHWC) -> (NCHW)

        # TODO: Classifier-Free Guidance (conditioning dropout)
        encoder_hidden_states, original_image_embeds = apply_conditioning_dropout(encoder_hidden_states, original_image_embeds, dropout_rng, bsz, args)

        # Concatenate the noisy latents with the original image embeddings
        concatenated_noisy_latents = jnp.concatenate([noisy_latents, original_image_embeds], axis=1)

        # Predict the noise residual and compute loss
        model_pred = unet.apply(
            {"params": params}, concatenated_noisy_latents, timesteps, encoder_hidden_states, train=True
        ).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = (target - model_pred) ** 2
        loss = loss.mean()

        return loss

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    # Update model parameters
    new_state = state.apply_gradients(grads=grad)
    
    # Apply EMA update 
    decay = get_decay(state.step)
    new_ema_params = ema_update(new_state.params, state.ema_params, decay)

    new_state = new_state.replace(ema_params=new_ema_params) 

    metrics = {"loss": loss}
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    return new_state, metrics, new_train_rng


# Create parallel version of the train step
p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

# Replicate the train state on each device
state = jax_utils.replicate(state)
text_encoder_params = jax_utils.replicate(text_encoder.params)
vae_params = jax_utils.replicate(vae_params)

# Train!
len_train_dataset =  train_dataloader.dataset_len

num_update_steps_per_epoch = math.ceil(len_train_dataset)

# Scheduler and math around the number of training steps.
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

#args.max_train_steps = math.ceil(args.max_train_steps*(4/total_train_batch_size))
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
logger.info("***** Running training *****")
logger.info(f"  Num examples = {len_train_dataset}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")



global_step = 0
epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
for epoch in epochs:
    # ======================== Training ================================
    train_metrics = []

    steps_per_epoch = len_train_dataset // total_train_batch_size
    train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
    # train
    for batch in train_dataloader:
        batch = shard(batch)
        state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
        train_metrics.append(train_metric)

        train_step_progress_bar.update(1)

        global_step += 1
        if global_step >= args.max_train_steps:
            break

    train_metric = jax_utils.unreplicate(train_metric)

    train_step_progress_bar.close()
    epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")


# Save trained model

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))

# Create the pipeline using using the trained modules and save it.
if jax.process_index() == 0:
    scheduler = FlaxPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
    )
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

    pipeline.save_pretrained(
        args.output_dir,
        params={
            "text_encoder": get_params_to_save(text_encoder_params),
            "vae": get_params_to_save(vae_params),
            "unet": get_params_to_save(state.ema_params),
            # "ema": get_params_to_save(state.ema_params),    
            # "tx": get_params_to_save(state.tx),
            "safety_checker": safety_checker.params,
        },
    )


# %%
