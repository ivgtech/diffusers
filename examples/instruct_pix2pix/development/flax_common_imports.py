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
from flax.training.common_utils import shard
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
    AutoencoderKL,
    UNet2DConditionModel,
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
from flax_from_hf_pretrained_april4 import get_pretrained
from model_converter import load_from_standard_weights
# from preprocess_load_data_april4 import train_dataset, train_dataloader, plot_batch

DEBUG = False 


