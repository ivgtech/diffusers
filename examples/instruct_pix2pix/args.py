import os
import math
import copy
import argparse
import logging
import requests
import random
import pandas as pd
from pathlib import Path
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from pickle import UnpicklingError
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

import io
import pyarrow.parquet as pq
import tensorflow as tf
import tensorflow_io as tfio

import torch
from torchvision import transforms
from torch.utils import data, checkpoint
from torch.utils.data import DataLoader

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

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = logging.getLogger(__name__)

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
    "timbrooks/instructpix2pix-clip-filtered": ("original_image", "edit_prompt", "edited_image"),
}

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

# convert the namespace to a dictionary
args = {
'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
'revision': 'flax',
'variant': None,
'dataset_name': 'timbrooks/instructpix2pix-clip-filtered', # Size of the auto-converted Parquet files: 130 GB. Number of rows: 313,010. Columns: original_image, edit_prompt, edited_image
'dataset_name': 'fusing/instructpix2pix-1000-samples',
'streaming': False,
'parquet': False,
'dataset_variant': None,
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

NUM_DEVICES = jax.device_count()
DEVICE_TYPE = jax.devices()[0].device_kind # 'TPU v4' etc.
TOTAL_TRAIN_BATCH_SIZE = args.train_batch_size * jax.device_count()
TRAIN_BATCH_SIZE = args.train_batch_size
MAX_TRAIN_STEPS = args.max_train_steps
NUM_TRAIN_EPOCHS = args.num_train_epochs