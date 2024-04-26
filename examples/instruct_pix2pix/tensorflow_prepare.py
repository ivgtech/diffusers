#  %% 

import jax
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
import io
from PIL import Image
import matplotlib.pyplot as plt

# Assuming necessary imports and configurations are already done

def read_dataset_from_parquet(parquet_file):
    table = pq.read_table(parquet_file)
    data = table.to_pydict()
    return data

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize((256, 256))  # Resize to the desired resolution
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image.astype(np.float32)

def create_tf_dataset(data):
    # Convert data to TensorFlow Dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    return tf_dataset

def prepare_data_for_tpu(tf_dataset, batch_size=16):
    # Preprocess and batch the data
    def _preprocess_features(original_image, edited_image, edit_prompt):
        return {
            "original_image": preprocess_image(original_image),
            "edited_image": preprocess_image(edited_image),
            "edit_prompt": edit_prompt
        }

    tf_dataset = tf_dataset.map(_preprocess_features)
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset

# Example usage

parquet_file = 'data/train-00000-of-00262-57cebf95b4a9170c.parquet'
data = read_dataset_from_parquet(parquet_file)

# Assuming 'original_image', 'edited_image', and 'edit_prompt' are keys in the parquet data
tf_dataset = create_tf_dataset({
    "original_image": data['original_image'],
    "edited_image": data['edited_image'],
    "edit_prompt": data['edit_prompt']
})

tf_dataset = prepare_data_for_tpu(tf_dataset)

# Example to visualize the data
for batch in tf_dataset.take(1):
    original_images = batch['original_image']
    edited_images = batch['edited_image']
    prompts = batch['edit_prompt']

    plt.figure(figsize=(10, 5))
    for i in range(2):  # Display first two images and prompts
        plt.subplot(1, 2, 1)
        plt.imshow(original_images[i])
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(edited_images[i])
        plt.title(f"Edited Image\nPrompt: {prompts[i]}")
        plt.axis('off')
    plt.show()
# %%
