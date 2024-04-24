import argparse
import logging
import math
import os
import sys
import random
import requests 
from pathlib import Path
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from pickle import UnpicklingError
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
from torch.utils import data
from torch.utils.data import DataLoader

import transformers
from datasets import load_dataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = logging.getLogger(__name__)

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]




class Args():
    def __init__(self, **kwargs):
      self.__dict__.update(kwargs)
      
args = {
    "pretrained_model_name_or_path": "timbrooks/instruct-pix2pix",
    "dataset_name": "fusing/instructpix2pix-1000-samples",
    "dataset_config_name": None,
    "train_data_dir": None,
    "cache_dir": None,
    "max_train_samples": None,
    "original_image_column": None,
    "edit_prompt_column": None,
    "edited_image_column": None,
    "resolution": 256,
    "center_crop": False,
    "random_flip": False,
    "train_batch_size": 1,
    "dataloader_num_workers": 0,
    "variant": None,
    "non_ema_revision": None,
    "report_to": "wandb",
    "hub_token": None,
    "seed": 42,
}
args = Args(**args)

if args.report_to == "wandb" and args.hub_token is not None:
    raise ValueError(
        "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
        " Please use `huggingface-cli login` to authenticate with the Hub."
    )

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# Setup logging, we only want one process per machine to log things on the screen.
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
if jax.process_index() == 0:
    transformers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()

if args.seed is not None:
    set_seed(args.seed)


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

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


##################################################################################################################################################################################
# DATA LOADERS 
##################################################################################################################################################################################

tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder='tokenizer', dtype=jnp.bfloat16)
# tokenizer = CLIPTokenizer.from_pretrained('../flax_models/instruct-pix2pix/tokenizer/')


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
    )
else:
    data_files = {}
    if args.train_data_dir is not None:
        data_files["train"] = os.path.join(args.train_data_dir, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=args.cache_dir,
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
train_dataset = dataset["train"].with_transform(preprocess_train)


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


def collate_fn_jax(examples):
    np_original_pixel_values = np.stack([example["original_pixel_values"] for example in examples]).astype(np.float32)
    np_edited_pixel_values = np.stack([example["edited_pixel_values"] for example in examples]).astype(np.float32)
    np_input_ids = np.stack([example["input_ids"] for example in examples])

    # Move data to accelerator
    original_pixel_values  = jax.device_put(np_original_pixel_values)
    edited_pixel_values = jax.device_put(np_edited_pixel_values)
    input_ids = jax.device_put(np_input_ids)

    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }


# DataLoaders creation:
train_dataloader_torch = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=args.train_batch_size,
    num_workers=args.dataloader_num_workers,
)

def torch_to_numpy(batch):
    # Convert a batch of PyTorch tensors to NumPy arrays.
    return {k: v.numpy() for k, v in batch.items()}

class JAXDataLoader:
    """
    training_generator = JAXDataLoader(train_dataloader_torch)

    for epoch in range(num_epochs):
        start_time = time.time()
        for batch in training_generator:
            images, labels = batch["images"], batch["labels"]
            labels = one_hot(labels, n_targets)  # Convert labels to one-hot encoding
            params = update(params, images, labels)  # Update model parameters
        epoch_time = time.time() - start_time
        # Compute accuracy on training and test datasets
    """
    def __init__(self, dataloader, batch_size=1, num_devices=1):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.num_devices = num_devices

    def __iter__(self):
        for batch in self.dataloader:   
            # Convert PyTorch tensors in the batch to NumPy arrays
            yield torch_to_numpy(batch)


class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_jax,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)



            
"""
import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

# Define our dataset, using torch datasets
# mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
# training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

"""




def plot_batch(sample, tokenizer):
    '''
    dict_keys(['original_pixel_values', 'edited_pixel_values', 'input_ids'])
    Shapes:(1, 3, 256, 256), (1, 3, 256, 256), (1, 77)
    '''

    # test dataloader by loading a batch and displaying the input_image, edit_prompt and edited_image
    # batch = next(iter(data_loader))
    original_images = sample["original_pixel_values"]
    edited_images = sample["edited_pixel_values"]
    captions = tokenizer.batch_decode(sample["input_ids"], skip_special_tokens=True)
    for original_image, edited_image, caption in zip(original_images, edited_images, captions):
        # is original_image object a PIL image?
        if isinstance(original_image, PIL.Image.Image):
            original_image = (original_image.permute(1, 2, 0) + 1) / 2
        else:
            original_image = (original_image.transpose(1, 2, 0) + 1) / 2
            
        if isinstance(edited_image, PIL.Image.Image):
            edited_image = (edited_image.permute(1, 2, 0) + 1) / 2
        else:
            edited_image = (edited_image.transpose(1, 2, 0) + 1) / 2
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.axis("off")
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(edited_image)
        plt.axis("off")
        plt.title(f'Prompt:"{caption}"')
        plt.show()


def batch_to_pil_plus_text(batch, tokenizer):
    original_images = batch["original_pixel_values"]
    edited_images = batch["edited_pixel_values"]
    captions = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    org_numpy_images, ed_numpy_images, texts = [], [], [] 
    for original_image, edited_image, caption in zip(original_images, edited_images, captions):
        # is original_image object a PIL image?
        if isinstance(original_image, PIL.Image.Image):
            original_image = (original_image.permute(1, 2, 0) + 1) / 2
        else:
            original_image = (original_image.transpose(1, 2, 0) + 1) / 2
            
        if isinstance(edited_image, PIL.Image.Image):
            edited_image = (edited_image.permute(1, 2, 0) + 1) / 2
        else:
            edited_image = (edited_image.transpose(1, 2, 0) + 1) / 2
        org_numpy_images.append(original_image)
        ed_numpy_images.append(edited_image)
        texts.append(caption)
    
    op_images = numpy_to_pil(np.array(org_numpy_images))
    ed_images = numpy_to_pil(np.array(ed_numpy_images))
    return op_images, ed_images, texts

def xbatch_to_pil_plus_text(batch, tokenizer):
    original_images = batch["original_pixel_values"]
    edited_images = batch["edited_pixel_values"]

    # (batch_size, 3, 256, 256) -> (batch_size, 256, 256, 3)
    original_images = np.transpose(original_images, (0, 2, 3, 1))
    edited_images = np.transpose(edited_images, (0, 2, 3, 1))

    # denormalize to [0, 1] from [-1, 1]
    original_images = (original_images + 1) / 2 
    edited_images = (edited_images + 1) / 2

    captions = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    return numpy_to_pil(original_images), numpy_to_pil(edited_images), captions


show_batch =  lambda sample: plot_batch(sample, tokenizer)

