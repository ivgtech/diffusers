# /diffusers/src/diffusers/models/modeling_flax_utils.py

import os
import sys
import torch

from typing import Any, Dict
from model_converter import load_from_standard_weights

import os
from pickle import UnpicklingError
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import create_repo, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_hf_hub_args,
)


from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    WEIGHTS_NAME,
    PushToHubMixin,
    logging,
)
logger = logging.get_logger(__name__)

from transformers import FlaxCLIPTextModel, CLIPTokenizer
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxDDPMScheduler

from diffusers.models.modeling_utils import load_state_dict 

import numpy as np
import jax.numpy as jnp
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import FlaxCLIPTextModel, CLIPTokenizer
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel, FlaxDDPMScheduler
from diffusers.models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from diffusers.models.modeling_utils import load_state_dict

 
 
def get_pretrained(pretrained_model_name_or_path: str, subfolder: str, cls: Any, force_download: bool = False) -> Dict:

    def create_flax_config(pretrained_model_name_or_path: str, subfolder: str, cls: Any) -> Dict:
        config, unused_kwargs = cls.load_config(
            pretrained_model_name_or_path,
            cache_dir=None,
            return_unused_kwargs=True,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            token=None,
            revision=None,
            subfolder=subfolder,
        )
        return config, unused_kwargs

    # cls = FlaxUNet2DConditionModel
    # pretrained_model_name_or_path = "timbrooks/instruct-pix2pix"

    config, unused_kwargs = create_flax_config(pretrained_model_name_or_path, subfolder, cls)

    model, model_kwargs = cls.from_config(config, dtype=jnp.bfloat16, return_unused_kwargs=True, **unused_kwargs)

    from_pt = True
    model_file = hf_hub_download(
        pretrained_model_name_or_path,
        filename=FLAX_WEIGHTS_NAME if not from_pt else WEIGHTS_NAME,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        user_agent=None,
        subfolder=subfolder,
        revision=None,
    )


    if from_pt:

        # Step 1: Get the pytorch file
        pytorch_model_file = load_state_dict(model_file)

        # Step 2: Convert the weights
        state = convert_pytorch_state_dict_to_flax(pytorch_model_file, model)
    else:
        try:
            with open(model_file, "rb") as state_f:
                state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            try:
                with open(model_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise EnvironmentError(f"Unable to convert {model_file} to Flax deserializable object. ")
        # make sure all arrays are stored as jnp.ndarray
        # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
        # https://github.com/google/flax/issues/1261
    state = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.local_devices(backend="cpu")[0]), state)

    # flatten dicts
    state = flatten_dict(state)

    params_shape_tree = jax.eval_shape(model.init_weights, rng=jax.random.PRNGKey(0))
    required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())

    shape_state = flatten_dict(unfreeze(params_shape_tree))

    missing_keys = required_params - set(state.keys())
    unexpected_keys = set(state.keys()) - required_params

    if missing_keys:
        logger.warning(
            f"The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. "
            "Make sure to call model.init_weights to initialize the missing weights."
        )
        cls._missing_keys = missing_keys

    for key in state.keys():
        if key in shape_state and state[key].shape != shape_state[key].shape:
            raise ValueError(
                f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
                f"{state[key].shape} which is incompatible with the model shape {shape_state[key].shape}. "
            )

    # remove unexpected keys to not be saved again
    for unexpected_key in unexpected_keys:
        del state[unexpected_key]

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
            " with another architecture."
        )
    else:
        logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    else:
        logger.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
            f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
            " training."
        )

    return model, unflatten_dict(state)