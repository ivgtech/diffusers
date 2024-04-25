# %%
import os
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from PIL import Image
import io
from transformers import CLIPTokenizer

# Load data from Parquet file
parquet_file = 'data/train-00000-of-00262-57cebf95b4a9170c.parquet'
df = pd.read_parquet(parquet_file, columns=["original_image", "edited_image", "edit_prompt"])
filename = './data/dataset.memmap'
# Initialize the tokenizer for CLIPTokenizer
#tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("timbrooks/instruct-pix2pix", subfolder='tokenizer')

def preprocess_images(image_dict):
    # Extract bytes from dictionary
    image_bytes = image_dict['bytes']
    # Open the image and preprocess
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((512, 512))  # Resize to 512x512
    image_np = np.array(image).astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1]
    return image_np

def preprocess_data(row):
    original_image = preprocess_images(row['original_image'])
    edited_image = preprocess_images(row['edited_image'])
    input_ids = tokenizer(row['edit_prompt'], return_tensors="np", max_length=77, padding="max_length", truncation=True)['input_ids'][0]
    return (original_image, edited_image, input_ids)


if not os.path.exists(filename):
    # Apply preprocessing
    processed_data = df.apply(preprocess_data, axis=1)

    # Save to binary file using memmap
    dtype = np.dtype([
        ('original_pixels', np.float32, (512, 512, 3)),
        ('edited_pixels', np.float32, (512, 512, 3)),
        ('input_ids', np.int32, (77,))
    ])

    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=(len(processed_data),))
    memmap_array[:] = list(processed_data)
    memmap_array.flush()

# %%
# Data loading function

def load_dataset(filename, batch_size):
    dtype = np.dtype([
        ('original_pixels', np.float32, (512, 512, 3)),
        ('edited_pixels', np.float32, (512, 512, 3)),
        ('input_ids', np.int32, (77,))
    ])
    data = np.memmap(filename, dtype=dtype, mode='r')
    num_batches = len(data) // batch_size
    
    for i in range(num_batches):
        batch = data[i * batch_size: (i + 1) * batch_size]
        yield collate_fn_jax(batch)

def collate_fn_jax(batch):
    original_images = [img['original_pixels'].transpose(2, 0, 1) for img in batch]
    edited_images = [img['edited_pixels'].transpose(2, 0, 1) for img in batch]
    input_ids = [ids['input_ids'] for ids in batch]

    np_original_pixel_values = np.stack(original_images)
    np_edited_pixel_values = np.stack(edited_images)
    padded_input_ids = np.stack(input_ids)

    original_pixel_values = jax.device_put(np_original_pixel_values)
    edited_pixel_values = jax.device_put(np_edited_pixel_values)
    input_ids = jax.device_put(padded_input_ids)

    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }
# %%

# Example usage
batch_size = 4 
for i, batch in enumerate(load_dataset(filename, batch_size)):
    # print(batch['original_pixel_values'].shape)
    # print(batch['edited_pixel_values'].shape)
    # print(batch['input_ids'].shape)
    # print(type(batch['original_pixel_values']))
    if i == 296:
      print(print(batch['original_pixel_values'].shape))
      print(batch['edited_pixel_values'].shape)
      print(batch['input_ids'].shape)
    pass
print(i)


# %%
