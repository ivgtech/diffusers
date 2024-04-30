# Effects of EMA Training on Diffusion Model Fine-Tuning Performance

InstructPix2Pix is a method to fine-tune text-conditioned diffusion models such that they can follow an edit instruction for an input image.

Despite substantial progress in open-source diffusion implementations, training and fine-tuning such
models continues to present significant challenges. 
The authors of the InstructPix2Pix paper fine-tuned Stable Diffusion 1.5 on a large dataset using 8x40GB A100 GPUs and over 25.5 hours.

Training diffusion models from scratch on the other hand, is prohibativly expensive:
* The original Stable Diffusion 2 project cost $600,000 to train (using 150,000 GPU-hours). 
* In 2023, MosaicML made substantial performance improvements and was able to bring that figure down to $50,000.


This project aims to replicate and extend the InstructPix2Pix model by implementing it in JAX,
utilizing Google Cloud TPU VMs in an effort to democratize access to diffusion models.

We aim to follow MosaicML's Exponential Moving Average (EMA) strategy, which applies EMA
only in the final stages of training.

This technique is expected to significantly reduce the computational overhead without compromising model quality, addressing a common bottleneck in diffusion model training.



***Disclaimer: The code of this implementation is based on two prior works:
(1) the original implementation by Brooks et. al. and (2) Hugging Face's InstructPix2Pix training example. The original code of the authors can be found [here](https://github.com/timothybrooks/instruct-pix2pix).
The Hugging Face Diffusers code can be found [here](https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix).***



**___Note: The flax example doesn't yet support features like gradient checkpoint, gradient accumulation etc, so to use flax for faster training we will need >30GB cards or TPU v3.
However, to replicate the paper using the original training dataset (310GB) it is strongly suggested you use a v4-32 TPU slice___**


## A few TPU preliminaries

Training Large Models on Cloud TPU requires first setting up a TPU VM.
This is fairly straightforward and well covered in Google's excellent TPU documentation. See the [run-calculation example](https://cloud.google.com/tpu/docs/run-calculation-jax) for an introduction.

A typical JAX training workflow usually means repeating the following steps with little to no variation:
1. spin up a TPU VM in a given zone
2. ssh into all workers and install JAX and any other software you may need such as `flax`, `diffusers`
3. download datasets and configuration files or (more likely) upload both to Google Cloud Storage
4. train your model
5. wait 
6. save the model weights and other artifacts to cloud storage
7. delete your Cloud TPU

Much of this is boilerplate, so opting to use a framework is often a good idea.
Here are a few:

* [Hugging Face Transformers](https://github.com/huggingface/transformers) or 
[Diffusers](https://github.com/huggingface/diffusers)
* [Levanter](https://github.com/stanford-crfm/levanter)
* [EasyLM]( https://github.com/young-geng/EasyLM)
* [JAXSeq]( https://github.com/Sea-Snell/JAXSeq) 
* [Mesh Transformer Jax]( https://github.com/kingoflolz/mesh-transformer-jax)
* [Paxml (aka Pax)](https://github.com/google/paxml)
* [FLAX](https://github.com/google/flax/tree/main/examples/lm1b)




## Sans Framework
Occasionally using a framework is not the best choice. Without one, we need to perform a few additional steps. As JAX and the entire TPU ecosystem are very tightly integrated this can be done fairly easily with a few bash commands or scripts.

First, we will need to export all the variables the `gcloud` CLI or parallel shell utility  expects. ***Note: TPU names should be lower case***

```
export PROJECT_ID= ...
export TPU_NAME= ...
export ZONE="us-central2-b"
export TPU_TYPE="v4-32"
export VERSION="tpu-ubuntu2204-base"
export STARTUP_SCRIPT="setup.sh"
```

after which we can spin up a preemptible machine:
```
"gcloud compute tpus queued-resources create ${TPU_NAME} \
    --node-id=${QR_ID} \
    --zone=${ZONE} \
    --accelerator-type=${TPU_TYPE} \
    --runtime-version=${VERSION}\
    --best-effort"     
```

Next, add the current VM's ssh keys:
```
ssh-add ~/.ssh/google_compute_engine
```

And finally, we can install software or perform other tasks using scripts or from the command line as shown below.
For example, to install Transformers:
```
gcloud compute tpus tpu-vm ssh ${VM_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command='git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    pip install . && \
    pip install -r examples/flax/_tests_requirements.txt && \
    pip install --upgrade huggingface-hub urllib3 zipp && \
    pip install tensorflow && \
    pip install -r examples/flax/language-modeling/requirements.txt'
```
or to download a checkpoint from Google Storage:

```
gcloud compute tpus tpu-vm ssh ${VM_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command='cd transformers/examples/flax/language-modeling && \
    gsutil cp -r gs://cloud-tpu-tpuvm-artifacts/v5litepod-preview/jax/gpt .'

```
## Using this code with other JAX/ Flax frameworks
Diffusers is the most popular library for state-of-the-art pretrained diffusion models in use currently. As such, we've modified our base JAX code to work seamlessly with it and to take advantage of the larger Hugging Face ecosystem. As most ML code is written with PyTorch and for NVIDIA GPUs, we felt that more people would
find and use our JAX code if it was integrated into the Diffusers framework.

Note that this codebase does not rely on Hugging Face. It can be modified to work as a stand-alone JAX library.
As JAX is generally quite agnostic to frameworks, using our training code with other JAX models or optimizers should be trivial.
Simply swap out any of the HF the models:
```
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
```
and replace them with your own. As long as the dimensions match everything will just work.

The training code in `flax_train_instruct_pix2pix.py` follows the Diffusers Flax training template and is just the usual JAX training loop.
Using it with other JAX code should be just a matter of cutting and pasting.
Following Diffusers convention, our inference code is found in the pipeline object (`src/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion_instruct_pix2pix.py`). Our `FlaxStableDiffusionInstructPix2PixPipeline` is essentially a wrapper for our JAX inference code.
However, the InstructPix2Pix model's inference loop is slightly more complicated than a standard diffusion models, and further needed to be integrating into the Pipeline Class, so extracting it might require more work than the training loop. 

As implemented in PyTorch, EMA smoothing,  usually involves some kind of secondary EMAModel class.
Our EMA implementiation is approximaely 30 lines of JAX code. It is integrated into the training loop but can be easily removed or modified.
Lastly, we implemented conditioning dropout for training as a standalone function.

Dataloading relies on the aria2 linux utility for downloading large files in parallel.
Our script should work with any dataset split into parquet files and available online.
To download `https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered` we use a python script to pull the full list of urls, shard the list of data files between all available devices and then write unique download texts assigning each machine a portion of the full dataset.
Next, we run a bash script to download all shards to their respecitve machine using aria2. The entire download takes approximaely 15 minutes.


We explored several options when it came to dataloading and processing. 
Hugging Face datasets library (streaming datasets specifically) were found to be easiest to use and integrate into our workflow but the IterableDataset class
was found to be quite slow compared to the alternatives.
PyTorch dataset streaming did not integrate well with the Hugging Face datasets class.
TensorFlow datasets were found to be excellent in terms of perormance but mimicking Pytorch transforms and intergrating TFDS with HF's streaming was problematic as was debugging TF lazy evaluation errors in a jittted/ distributed environment.
To make fix that we explored downloading the original dataset, processing into TFDS compatible formats, and then uploading it to Google Storage. However, that was considered to be too timeconsuming and also not an approach that would scale well (other datasets, and much larger datasets). 
Lastly, potential costs associated with Cloud storage might be an accesability issue for many potential users as data movement costs can be considerable with very large datasets such as ours.
Our compromise solution was to write our own dataloading code, leveraging PyTorch's data classes, Linux utilites and JAX/ XLA's data parallelism.
Data processing is done on the fly using `pareque_preprocessing.py`.  
Dataloading is done with the same file and under-the-hood relies on a combinatino of PyTorch DataSet classes and our JAX numpy code.


Checkpointing is not used. We know from past experience that reloading a checkpoint after a crash may take up to several hours when using remote storage for datasets and model files.
Furthermore, as this was an EMA model,  recovering weights, paramaters and EMA parameters would certaintly make this process longer.
As our total training run was 12 to 24 hours, and as TPU training is incredibly stable, we felt it best to avoid, rather than
risk 1 or 2 hours out of a  12 hour test training run to recover potentially unsuable checkpoints.
Similar considerations about fast training runs meant we did not log validation images to wandb.



## Replicating the paper

### TPU setup

Create a TPU project as described in [Set up an account and Cloud TPU project](https://cloud.google.com/tpu/docs/setup-gcp-account) and then create a `v4-32` Pod slice using the gcloud command.

### Installing the dependencies

Use the setup-tpu-vm.sh script to install JAX and all the required dependencies. ***Recall that scripts need to be run in parallel across all devices which means using ssh with `gcloud` or `pdsh` etc.***

### Distributed dataset downloading and processing
To replicate the InstructPix2Pix paper we'll use the [original dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) used by the authors for training. 
To compare our implementation to Hugging Face's, we'll also use their [small toy dataset](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples), which is a smaller version of the original. 

Run `script.py` to create the download.txt that you'll use with your bash script to download and shard the dataset across you Pod slice.

Run `script.sh` to download and shard the full dataset to each machine.

### Create a modified Flax Unet
The InstructPix2Pix uses a unet with 8 input channels. The Diffusers Flax unet model class uses 4 channels. In PyTorch changing the number of channels is trivial but more complicated in JAX because of the way JAX/ Flax instantiates models.

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=us-central2-b --worker=all --command="source venv/bin/activate && git clone https://github.com/baricev/diffusers.git && cd diffusers && pip install -e '.[dev]' "

Run create_modified_unet.py to create and save the unet model used for training.
```
 gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=us-central2-b --worker=all \
                                --command="source /home/v/venv/bin/activate \
                                cd research_projects/instructpix2pix_flax && \
                                python convert.py"
```

```
 gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=us-central2-b --worker=all \
                                --command="source /home/v/venv/bin/activate && \
                                cd research_projects/instructpix2pix_flax && \
                                python .py"
```


### Toy example
To run the toy example (5 hours using a A100 40GB) takes only 5 minutes.

Change the `train_downloader` and run the training script:
```
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=us-central2-b \
    --worker=all 
    --command="source venv/bin/activate && \
          cd diffusers/examples/instruct_pix2pix && \
          python flax_train_instruct_pix2pix.py"
  ```
### Full datatset example
Change the `train_downloader` and `parquet_processing.py` script to use the full dataset.

You can also train on a portion of the full dataset by modifying the `dataset_size` in 
`parquet_processing.py`.
```
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=us-central2-b \
    --worker=all 
    --command="source venv/bin/activate && \
          cd diffusers/examples/instruct_pix2pix && \
          python flax_train_instruct_pix2pix.py"
  ```







Refer to the original InstructPix2Pix training example for installing the dependencies.

You will also need to get access of SDXL by filling the [form](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). 

### Toy example

As mentioned before, we'll use a [small toy dataset](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples) for training. The dataset 
is a smaller version of the [original dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) used in the InstructPix2Pix paper.

Configure environment variables such as the dataset identifier and the Stable Diffusion
checkpoint:

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_ID="fusing/instructpix2pix-1000-samples"
```

Now, we can launch training:

```bash
accelerate launch train_instruct_pix2pix_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --push_to_hub
```

Additionally, we support performing validation inference to monitor training progress
with Weights and Biases. You can enable this feature with `report_to="wandb"`:

```bash
accelerate launch train_instruct_pix2pix_sdxl.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --dataset_name=$DATASET_ID \
    --use_ema \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --val_image_url_or_path="https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg" \
    --validation_prompt="make it in japan" \
    --report_to=wandb \
    --push_to_hub
 ```

 We recommend this type of validation as it can be useful for model debugging. Note that you need `wandb` installed to use this. You can install `wandb` by running `pip install wandb`. 

 [Here](https://wandb.ai/sayakpaul/instruct-pix2pix-sdxl-new/runs/sw53gxmc), you can find an example training run that includes some validation samples and the training hyperparameters.

 ***Note: In the original paper, the authors observed that even when the model is trained with an image resolution of 256x256, it generalizes well to bigger resolutions such as 512x512. This is likely because of the larger dataset they used during training.***

 ## Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. Follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash 
accelerate launch --mixed_precision="fp16" --multi_gpu train_instruct_pix2pix_sdxl.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --dataset_name=$DATASET_ID \
    --use_ema \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --val_image_url_or_path="https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg" \
    --validation_prompt="make it in japan" \
    --report_to=wandb \
    --push_to_hub
```

 ## Inference

 Once training is complete, we can perform inference:

 ```python
import PIL
import requests
import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline

model_id = "your_model_id" # <- replace this 
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

url = "https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg"


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

image = download_image(url)
prompt = "make it Japan"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(prompt, 
    image=image, 
    num_inference_steps=num_inference_steps, 
    image_guidance_scale=image_guidance_scale, 
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]
edited_image.save("edited_image.png")
```

We encourage you to play with the following three parameters to control
speed and quality during performance:

* `num_inference_steps`
* `image_guidance_scale`
* `guidance_scale`

Particularly, `image_guidance_scale` and `guidance_scale` can have a profound impact
on the generated ("edited") image (see [here](https://twitter.com/RisingSayak/status/1628392199196151808?s=20) for an example).

If you're looking for some interesting ways to use the InstructPix2Pix training methodology, we welcome you to check out this blog post: [Instruction-tuning Stable Diffusion with InstructPix2Pix](https://huggingface.co/blog/instruction-tuning-sd). 

## Compare between SD and SDXL

We aim to understand the differences resulting from the use of SD-1.5 and SDXL-0.9 as pretrained models. To achieve this, we trained on the [small toy dataset](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples) using both of these pretrained models. The training script is as follows:

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5" or "stabilityai/stable-diffusion-xl-base-0.9"
export DATASET_ID="fusing/instructpix2pix-1000-samples"

accelerate launch train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --use_ema \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --val_image_url="https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg" \
    --validation_prompt="make it in Japan" \
    --report_to=wandb \
    --push_to_hub
```

We discovered that compared to training with SD-1.5 as the pretrained model, SDXL-0.9 results in a lower training loss value (SD-1.5 yields 0.0599, SDXL scores 0.0254). Moreover, from a visual perspective, the results obtained using SDXL demonstrated fewer artifacts and a richer detail. Notably, SDXL starts to preserve the structure of the original image earlier on.

The following two GIFs provide intuitive visual results. We observed, for each step, what kind of results could be achieved using the image 
<p align="center">
    <img src="https://datasets-server.huggingface.co/assets/fusing/instructpix2pix-1000-samples/--/fusing--instructpix2pix-1000-samples/train/23/input_image/image.jpg" alt="input for make it Japan" width=600/>
</p>
with "make it in Japan” as the prompt. It can be seen that SDXL starts preserving the details of the original image earlier, resulting in higher fidelity outcomes sooner.

* SD-1.5: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sd_ip2p_training_val_img_progress.gif

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sd_ip2p_training_val_img_progress.gif" alt="input for make it Japan" width=600/>
</p>

* SDXL: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_ip2p_training_val_img_progress.gif

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl_ip2p_training



# Stable Diffusion XL for JAX + TPUv5e

[TPU v5e](https://cloud.google.com/blog/products/compute/how-cloud-tpu-v5e-accelerates-large-scale-ai-inference) is a new generation of TPUs from Google Cloud. It is the most cost-effective, versatile, and scalable Cloud TPU to date. This makes them ideal for serving and scaling large diffusion models.

[JAX](https://github.com/google/jax) is a high-performance numerical computation library that is well-suited to develop and deploy diffusion models:

- **High performance**. All JAX operations are implemented in terms of operations in [XLA](https://www.tensorflow.org/xla/) - the Accelerated Linear Algebra compiler

- **Compilation**. JAX uses just-in-time (jit) compilation of JAX Python functions so it can be executed efficiently in XLA. In order to get the best performance, we must use static shapes for jitted functions, this is because JAX transforms work by tracing a function and to determine its effect on inputs of a specific shape and type. When a new shape is introduced to an already compiled function, it retriggers compilation on the new shape, which can greatly reduce performance. **Note**: JIT compilation is particularly well-suited for text-to-image generation because all inputs and outputs (image input / output sizes) are static.

- **Parallelization**. Workloads can be scaled across multiple devices using JAX's [pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html), which expresses single-program multiple-data (SPMD) programs. Applying pmap to a function will compile a function with XLA, then execute in parallel on XLA devices. For text-to-image generation workloads this means that increasing the number of images rendered simultaneously is straightforward to implement and doesn't compromise performance.

👉 Try it out for yourself:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/google/sdxl)

## Stable Diffusion XL pipeline in JAX

Upon having access to a TPU VM (TPUs higher than version 3), you should first install
a TPU-compatible version of JAX:
```
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Next, we can install [flax](https://github.com/google/flax) and the diffusers library:

```
pip install flax diffusers transformers
```

In [sdxl_single.py](./sdxl_single.py) we give a simple example of how to write a text-to-image generation pipeline in JAX using [StabilityAI's Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0).

Let's explain it step-by-step:

**Imports and Setup**

```python
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionXLPipeline

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("/tmp/sdxl_cache")
import time

NUM_DEVICES = jax.device_count()
```

First, we import the necessary libraries:
- `jax` is provides the primitives for TPU operations
- `flax.jax_utils` contains some useful utility functions for `Flax`, a neural network library built on top of JAX
- `diffusers` has all the code that is relevant for SDXL.
- We also initialize a cache to speed up the JAX model compilation.
- We automatically determine the number of available TPU devices.

**1. Downloading Model and Loading Pipeline**

```python
pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", revision="refs/pr/95", split_head_dim=True
)
```
Here, a pre-trained model `stable-diffusion-xl-base-1.0` from the namespace `stabilityai` is loaded. It returns a pipeline for inference and its parameters.

**2. Casting Parameter Types**

```python
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state
```
This section adjusts the data types of the model parameters.
We convert all parameters to `bfloat16` to speed-up the computation with model weights.
**Note** that the scheduler parameters are **not** converted to `blfoat16` as the loss
in precision is degrading the pipeline's performance too significantly.

**3. Define Inputs to Pipeline**

```python
default_prompt = ...
default_neg_prompt = ...
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 25
```
Here, various default inputs for the pipeline are set, including the prompt, negative prompt, random seed, guidance scale, and the number of inference steps.

**4. Tokenizing Inputs**

```python
def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids
```
This function tokenizes the given prompts. It's essential because the text encoders of SDXL don't understand raw text; they work with numbers. Tokenization converts text to numbers.

**5. Parallelization and Replication**

```python
p_params = replicate(params)

def replicate_all(prompt_ids, neg_prompt_ids, seed):
    ...
```
To utilize JAX's parallel capabilities, the parameters and input tensors are duplicated across devices. The `replicate_all` function also ensures that every device produces a different image by creating a unique random seed for each device.

**6. Putting Everything Together**

```python
def generate(...):
    ...
```
This function integrates all the steps to produce the desired outputs from the model. It takes in prompts, tokenizes them, replicates them across devices, runs them through the pipeline, and converts the images to a format that's more interpretable (PIL format).

**7. Compilation Step**

```python
start = time.time()
print(f"Compiling ...")
generate(default_prompt, default_neg_prompt)
print(f"Compiled in {time.time() - start}")
```
The initial run of the `generate` function will be slow because JAX compiles the function during this call. By running it once here, subsequent calls will be much faster. This section measures and prints the compilation time.

**8. Fast Inference**

```python
start = time.time()
prompt = ...
neg_prompt = ...
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")
```
Now that the function is compiled, this section shows how to use it for fast inference. It measures and prints the inference time.

In summary, the code demonstrates how to load a pre-trained model using Flax and JAX, prepare it for inference, and run it efficiently using JAX's capabilities.

## Ahead of Time (AOT) Compilation

FlaxStableDiffusionXLPipeline takes care of parallelization across multiple devices using jit. Now let's build parallelization ourselves.

For this we will be using a JAX feature called [Ahead of Time](https://jax.readthedocs.io/en/latest/aot.html) (AOT) lowering and compilation. AOT allows to fully compile prior to execution time and have control over different parts of the compilation process.

In [sdxl_single_aot.py](./sdxl_single_aot.py) we give a simple example of how to write our own parallelization logic for text-to-image generation pipeline in JAX using [StabilityAI's Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0)

We add a `aot_compile` function that compiles the `pipeline._generate` function
telling JAX which input arguments are static, that is, arguments that
are known at compile time and won't change. In our case, it is num_inference_steps,
height, width and return_latents.

Once the function is compiled, these parameters are omitted from future calls and
cannot be changed without modifying the code and recompiling.

```python
def aot_compile(
        prompt=default_prompt,
        negative_prompt=default_neg_prompt,
        seed=default_seed,
        guidance_scale=default_guidance_scale,
        num_inference_steps=default_num_steps
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]

    return pmap(
        pipeline._generate,static_broadcasted_argnums=[3, 4, 5, 9]
        ).lower(
            prompt_ids,
            p_params,
            rng,
            num_inference_steps, # num_inference_steps
            height, # height
            width, # width
            g,
            None,
            neg_prompt_ids,
            False # return_latents
            ).compile()
````

Next we can compile the generate function by executing `aot_compile`.

```python
start = time.time()
print("Compiling ...")
p_generate = aot_compile()
print(f"Compiled in {time.time() - start}")
```
And again we put everything together in a `generate` function.

```python
def generate(
    prompt,
    negative_prompt,
    seed=default_seed,
    guidance_scale=default_guidance_scale
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(
        prompt_ids,
        p_params,
        rng,
        g,
        None,
        neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))
```

The first forward pass after AOT compilation still takes a while longer than
subsequent passes, this is because on the first pass, JAX uses Python dispatch, which
Fills the C++ dispatch cache.
When using jit, this extra step is done automatically, but when using AOT compilation,
it doesn't happen until the function call is made.

```python
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"First inference in {time.time() - start}")
```

From this point forward, any calls to generate should result in a faster inference
time and it won't change.

```python
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")
```
