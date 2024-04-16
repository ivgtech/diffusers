# %% 

import os
import copy
import random
import PIL
import math
import argparse
import logging
import requests
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax import jax_utils
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.training.common_utils import shard
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from flax import serialization # to_bytes, from_bytes
from flax.jax_utils import replicate, unreplicate

import torch
import torch.utils.checkpoint
from torchvision import transforms

import transformers
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed
from huggingface_hub import create_repo, upload_folder
from datasets import load_dataset

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version

# 
from jax_dataloader import NumpyLoader, train_dataset, show_batch, batch_to_pil_plus_text

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")
logger = logging.getLogger(__name__)


# convert the namespace to a dictionary
args = {
'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
'revision': 'flax',
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
'resolution': 256,
'center_crop': False,
'random_flip': True,
'train_batch_size': 4,
'num_train_epochs': 2, #100,
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
'ema_decay': 0.999,
'max_ema_decay': 0.999,
'min_ema_decay': 0.5,
'ema_decay_power': 0.6666666,
'ema_inv_gamma': 1.0,
'start_ema_update_after': 100,
'update_ema_every': 10,
'save_unet_and_ema_params': False,
'using_original_dataset': False,
# 'dataset_name': 'timbrooks/instructpix2pix-clip-filtered',
# 'original_dataset_length': 313010,
# 'original_max_epochs': 2000, # when using the full instruct-pix2-pix dataset
}

class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries) 

args = Args(**args)

USING_FULL_DATASET = args.using_original_dataset

if USING_FULL_DATASET:
    args.dataset_name = "timbrooks/instructpix2pix-clip-filtered"
    args.original_image_column = 'original_image',
    args.edited_image_column = 'edited_image',
    args.edit_prompt_column = 'edit_prompt',
    args.dataset_length = 313010
    args.max_epochs = 2000
else: 
    args.dataset_length = 1000

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


# Dataset 

tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder='tokenizer', 
    dtype=jnp.bfloat16
    )

# Get the datasets: you can either provide your own training and evaluation files (see below)
# or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

# In distributed training, the load_dataset function guarantees that only one local process can concurrently
# download the dataset.
if args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        streaming=True,
    )
else:
    data_files = {}
    if args.train_data_dir is not None:
        data_files["train"] = os.path.join(args.train_data_dir, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=args.cache_dir,
        streaming=True,
    )
    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

# Preprocessing the datasets.
# We need to tokenize inputs and targets.
column_names = dataset["train"].column_names

# 6. Get the column names for input/target.
dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
if args.original_image_column is None:
    original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
else:
    original_image_column = args.original_image_column
    if original_image_column not in column_names:
        raise ValueError(
            f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
        )
if args.edit_prompt_column is None:
    edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
else:
    edit_prompt_column = args.edit_prompt_column
    if edit_prompt_column not in column_names:
        raise ValueError(
            f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
        )
if args.edited_image_column is None:
    edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
else:
    edited_image_column = args.edited_image_column
    if edited_image_column not in column_names:
        raise ValueError(
            f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
        )


# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# Preprocessing the datasets.
train_transforms = transforms.Compose(
    [
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    ]
)

def preprocess_images(examples):
    original_images = np.concatenate(
        [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
    )
    edited_images = np.concatenate(
        [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
    )
    # We need to ensure that the original and the edited images undergo the same
    # augmentation transforms.
    images = np.concatenate([original_images, edited_images])
    images = torch.tensor(images)
    images = 2 * (images / 255) - 1
    return train_transforms(images)

def preprocess_train(examples):
    # Preprocess images.
    preprocessed_images = preprocess_images(examples)
    # Since the original and edited images were concatenated before
    # applying the transformations, we need to separate them and reshape
    # them accordingly.
    original_images, edited_images = preprocessed_images.chunk(2)
    original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
    edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

    # Collate the preprocessed images into the `examples`.
    examples["original_pixel_values"] = original_images
    examples["edited_pixel_values"] = edited_images

    # Preprocess the captions.
    captions = list(examples[edit_prompt_column])
    examples["input_ids"] = tokenize_captions(captions)
    return examples

if args.max_train_samples is not None:
    dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))

# Set the training transforms
# train_dataset = dataset["train"].with_transform(preprocess_train)

train_dataset = dataset["train"].map(preprocess_train, batched=True, batch_size=args.train_batch_size)


def collate_fn(examples):
    original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }


shuffled_dataset = train_dataset.shuffle(seed=42, buffer_size=1_000)

# DataLoaders creation:
train_dataloader_torch = torch.utils.data.DataLoader(
    shuffled_dataset,
    collate_fn=collate_fn,
    batch_size=args.train_batch_size,
    num_workers=args.dataloader_num_workers,
)

def torch_to_numpy(batch):
    return {k: v.numpy() for k, v in batch.items()}

class JAXDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
    def __iter__(self):
        for batch in self.dataloader:
            # Convert PyTorch tensors in the batch to NumPy arrays
            yield torch_to_numpy(batch)

# Helper functions

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

def NHWC_to_NCHW(tensor):
    return jnp.einsum("ijkl->iljk", tensor)

def retrieve_latents(**kwargs):
    return NHWC_to_NCHW(retrieve_latents_jax(**kwargs))

def curry_retrieve_and_transform(NHWC_to_NCHW_func):
    def retrieve_and_transform(**kwargs):
        latents = retrieve_latents_jax(**kwargs)
        return NHWC_to_NCHW_func(latents)
    return retrieve_and_transform


def get_nparams(params: FrozenDict) -> int:
    nparams = 0
    for item in params:
        if isinstance(params[item], FrozenDict) or isinstance(params[item], dict):
            nparams += get_nparams(params[item])
        else:
            nparams += params[item].size
    return nparams


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




class EMAState(struct.PyTreeNode):
    """ ema_params: A FrozenDict containing the EMA of the parameters.  """
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

class ExtendedTrainState(TrainState, EMAState):
    """ A simple train state for the common case with a single Optax optimizer that also tracks
    the exponential moving average (EMA) of the model parameters.
    This state is extended to handle both the raw model parameters and their EMA values for
    training and validation purposes, respectively.  """
    def apply_gradients( self, grads: core.FrozenDict[str, Any]) -> "ExtendedTrainState":
        """
        Apply gradients to parameters and update EMA parameters.
        Args:
            grads: Gradients computed during backpropagation to be applied to the parameters.
            ema_decay: The decay factor for the EMA parameters.
        """
        # Standard parameter update
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        # # Update EMA parameters
        new_ema_params = jax.tree.map(
            lambda ema, p: ema * args.ema_decay + (1 - args.ema_decay) * p, self.ema_params, new_params
        )
        return self.replace( params=new_params, opt_state=new_opt_state, ema_params=new_ema_params)

# class ExtendedTrainState(TrainState, EMAState):
#     """
#     Extended train state to include EMA parameters and dynamic EMA updates.
#     """
#     def apply_gradients(self, grads: core.FrozenDict[str, Any], timestep: int, max_ema_decay: float = 0.999,
#                         min_ema_decay: float = 0.5, ema_decay_power: float = 0.6666666, ema_inv_gamma: float = 1.0,
#                         start_ema_update_after: int = 100, update_ema_every: int = 10) -> "ExtendedTrainState":
#         """
#         Apply gradients and update EMA parameters dynamically based on training steps.

#         Args:
#             grads: Gradients to apply.
#             timestep: Current step in training used to compute the EMA decay rate dynamically.
#             max_ema_decay: Maximum decay rate.
#             min_ema_decay: Minimum decay rate.
#             ema_decay_power: Decay power factor.
#             ema_inv_gamma: Inverse gamma for decay calculation.
#             start_ema_update_after: Number of steps after which EMA updates start.
#             update_ema_every: Frequency of EMA updates in number of steps.

#         Returns:
#             Updated `ExtendedTrainState`.
#         """
#         # Apply the usual gradients to update parameters
#         updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
#         new_params = optax.apply_updates(self.params, updates)

#         # Update EMA parameters using the dynamic decay rate calculation
#         def calculate_decay():
#             decay = 1.0 - (1.0 + (timestep / ema_inv_gamma)) ** (-ema_decay_power)
#             return jnp.clip(decay, min_ema_decay, max_ema_decay)

#         if timestep >= start_ema_update_after and timestep % update_ema_every == 0:
#             decay = calculate_decay()
#             decay_avg = 1.0 - decay
#             new_ema_params = jax.tree_util.tree_map(
#                 lambda ema, p_new: decay_avg * ema + decay * p_new,
#                 self.ema_params,
#                 new_params,
#             )
#         else:
#             new_ema_params = self.ema_params

#        return self.replace(params=new_params, opt_state=new_opt_state, ema_params=new_ema_params)


# Training dataloader

# Creates a JAX dataloader from a standard PyTorch dataloader
train_dataloader = NumpyLoader(
    train_dataset, 
    batch_size=args.train_batch_size,
    num_workers= 0
)

# Models and state

weight_dtype = jnp.float32
if args.mixed_precision == "fp16":
    weight_dtype = jnp.float16
elif args.mixed_precision == "bf16":
    weight_dtype = jnp.bfloat16

assert weight_dtype == jnp.bfloat16, "Out of memory error with fp16. Please use bfloat16." 

# Load models and create wrapper for stable diffusion
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    from_pt=args.from_pt,
    revision=args.revision,
    subfolder="tokenizer",
)
text_encoder = FlaxCLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path,
    from_pt=args.from_pt,
    revision=args.revision,
    subfolder="text_encoder",
    dtype=weight_dtype,
)

# Load the vae-ft-mse-840000-ema-pruned.ckp
vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    '../flax_models/stable-diffusion-v1-5', 
    subfolder="vae",
    revision="flax", # non-EMA weights
    dtype=weight_dtype,
)

# Load the converted unet model
save_dir = 'modified_unet'
unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    save_dir,
    dtype=weight_dtype,
)


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
# Initialize EMA params with original model params
ema_params = copy.deepcopy(unet_params)
print("Number of EMA parameters: ", get_nparams(ema_params))

# Prepare optimizer and state, incMA parameters
state = ExtendedTrainState.create(apply_fn=unet.__call__, params=unet_params, ema_params=ema_params, tx=optimizer)
#state = TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

noise_scheduler = FlaxDDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
noise_scheduler_state = noise_scheduler.create_state()

# Initialize our training
rng = jax.random.PRNGKey(args.seed)
train_rngs = jax.random.split(rng, jax.local_device_count())

# print("Starting training...")
# return 0


def train_step(state, text_encoder_params, vae_params, batch, train_rng):
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

    def compute_loss(params):

        # Convert images to latent space
        vae_outputs = vae.apply(
            {"params": vae_params}, batch["edited_pixel_values"], deterministic=True, method=vae.encode
        )
        latents = vae_outputs.latent_dist.sample(sample_rng)
        # (NHWC) -> (NCHW)
        latents = jnp.einsum("ijkl->iljk", latents) * vae.config.scaling_factor
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
        # (NHWC) -> (NCHW)
        original_image_embeds = jnp.einsum("ijkl->iljk", original_image_embeds)

        # TODO: Implement conditioning dropout

        # Concatenate the `original_image_embeds` with the `noisy_latents`.
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

    new_state = state.apply_gradients(grads=grad)

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

# Assume streaming data so no len() available
try:  
    len_train_dataset = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
except:
    # Note: 'IterableDataset' objects have no len()
    len_train_dataset = args.dataset_length
    num_update_steps_per_epoch = args.dataset_length

# Scheduler and math around the number of training steps.
if args.using_original_dataset:
    num_update_steps_per_epoch = args.dataset_length
    args.num_train_epochs = args.max_epochs
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch 
else: 
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch # 60 * 250 = 15000

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch) # 15000 / 250 = 60

# args.num_train_epochs = 1  # NOTE: TESTING ONLY
# args.max_train_steps = 15000


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

# %%     
# Save trained weights

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

    args.save_non_ema_unet_params = False

    if args.save_non_ema_unet_params:
        logger.info('Saving Non-EMA unet parameters to disk.')
        # 'non-ema' parameters need to be saved separately as saving additional parameters through the save_pretrained method is not supported.
        non_ema_params_to_save = get_params_to_save(state.params)
        non_ema_params_path = os.path.join(args.output_dir, 'unet')
        non_ema_params_bytes = serialization.to_bytes(non_ema_params_to_save)
        with open(os.path.join(non_ema_params_path, 'non_ema.msgpack'), 'wb') as f:
            f.write(non_ema_params_bytes)

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

#if __name__ == "__main__":
# main()




# %%
