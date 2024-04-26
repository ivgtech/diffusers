#  %% 
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
import io
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed


# # Assuming necessary imports and configurations are already done

# def _read_dataset_from_parquet(parquet_file):
#     table = pq.read_table(parquet_file)
#     data = table.to_pydict()
#     return data

# def preprocess_image(image_bytes):
#     print(type(image_bytes))
#     image = Image.open(io.BytesIO(image_bytes))
#     #image = image.convert("RGB")  # Ensure RGB format
#     #image = image.resize((256, 256))  # Resize to the desired resolution
#     image = np.array(image) / 255.0  # Normalize to [0, 1]
#     return image.astype(np.float32)

# def create_tf_dataset(data):
#     # Convert data to TensorFlow Dataset
#     tf_dataset = tf.data.Dataset.from_tensor_slices(data)
#     return tf_dataset

# batch_size = jax.device_count() 

# def prepare_data_for_tpu(tf_dataset, batch_size=batch_size):
#     # Preprocess and batch the data
#     def _preprocess_features(original_image, edited_image, edit_prompt):
#         return {
#             "original_image": preprocess_image(original_image),
#             "edited_image": preprocess_image(edited_image),
#             "edit_prompt": edit_prompt
#         }

#     tf_dataset = tf_dataset.map(_preprocess_features)
#     #tf_dataset = tf_dataset.map(_preprocess_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     tf_dataset = tf_dataset.batch(batch_size)
#     #tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     return tf_dataset

# # Example usage

# parquet_file = 'data/train-00000-of-00262-57cebf95b4a9170c.parquet'
# data = read_dataset_from_parquet(parquet_file)



def read_dataset_from_parquet(parquet_file):
    # Load the parquet file
    table = pq.read_table(parquet_file)
    data = table.to_pydict()

    column_names = ["original_image", "edit_prompt", "edited_image"]

    for column_name in column_names:
        if column_name not in data:
            raise ValueError(f"Column '{column_name}' not found in the Parquet file.")

        # Read byte char array as PIL images and then convert to numpy array
        if 'image' in column_name:
            image_bytes_dict = data[column_name]

            bytes = [byte_data['bytes'] for byte_data in image_bytes_dict]
            file_names = [byte_data['path'] for byte_data in image_bytes_dict]

            # Keep images as PIL Images, for further processing rather than converting to numpy arrays immediately
            pil_images = [Image.open(io.BytesIO(byte_data)) for byte_data in bytes]
            # numpy_images = [np.array(img) for img in pil_images]
            data[column_name] = pil_images


    # drop all other columns
    data = {column_name: data[column_name] for column_name in column_names}

    return data

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", dtype=jnp.bfloat16) 

def tokenize_captions(captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length, 
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    return inputs.input_ids

def f(x):
    if isinstance(x,str):
        return tokenize_captions(x) 

    image = x.convert("RGB")  # Ensure RGB format
    image = image.resize((256, 256))  # Resize to the desired resolution
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image.astype(np.float32)

def create_tf_dataset(data):
    # Convert data to TensorFlow Dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    return tf_dataset  

pf = 'data/train-00000-of-00262-57cebf95b4a9170c.parquet'
xs = read_dataset_from_parquet(pf)
data = jax.tree.map(f, xs)

# %% 
# NOTE: tf.data.Dataset.from_tensor_slices divides up the dataset along the first or batch dimension
# So, we need to ensure that the data is already batched before passing it to tf.data.Dataset.from_tensor_slices
# otherwise the call essentially hangs

tf_dataset = create_tf_dataset({
    "original_image": np.array(data['original_image']),
    "edited_image": np.array(data['edited_image']),
    "edit_prompt": np.array(data['edit_prompt'])
})




# Define a function to convert tensors to numpy arrays
def to_numpy(original_image, edited_image, edit_prompt):
    return original_image.numpy(), edited_image.numpy(), edit_prompt.numpy()

def show_lazy_eval_images(N=4):

    xs = tf_dataset.take(N)

    # Use the map function to apply the conversion
    numpy_dataset = xs.map(lambda x: tf.py_function(
        to_numpy, 
        [x['original_image'], x['edited_image'], x['edit_prompt']], 
        [tf.float32, tf.float32, tf.int64]))  # Specify the correct output types

    # Convert to list of tuples (original_image, edited_image, edit_prompt)
    ls = list(numpy_dataset.as_numpy_iterator())

    first_row = ls[:N//2]
    second_row = ls[N//2:]
    prompt_list = [x[2] for x in first_row + second_row]
    prompt_list = [tokenizer.batch_decode(x, skip_special_tokens=True) for x in prompt_list]

    # Display the images

    fig, axs = plt.subplots(2, N,  figsize= (N*4, 8))
    # Set axis to '
    for i, ex in enumerate(ls):
        # for j, (original_image, edited_image, edit_prompt) in enumerate(ex):
            axs[0, i].imshow(ex[0])
            axs[1, i].imshow(ex[1])
            axs[0,i].axis('off')
            axs[1,i].axis('off')
            axs[1, i].set_title(f"Prompt: {prompt_list[i][0]}") 
    plt.show()  
    

# show_lazy_eval_images(10)


# %%
