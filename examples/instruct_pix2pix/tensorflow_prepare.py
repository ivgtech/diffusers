#  %% 
from args import *


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


def read_dataset_from_arrow(dataset_path, split='train'):
    # Load a Hugging Face dataset from the specified name or path and split. 
    # Example usage: xs = read_dataset_from_arrow('fusing/instructpix2pix-1000-samples')

    dataset = load_dataset(dataset_path, split=split)

    column_names = ["input_image", "edit_prompt", "edited_image"]

    # Verify if all required columns are present in the dataset
    for column_name in column_names:
        if column_name not in dataset.column_names:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")

    # Process image columns
    processed_data = {column_name: [] for column_name in column_names}
    for example in dataset:
        for column_name in column_names:
            if 'image' in column_name:
                # Check if the data is already a PIL Image
                image_data = example[column_name]
                if isinstance(image_data, Image.Image):
                    pil_image = image_data
                else:
                    # Convert image bytes to PIL Image if not already
                    pil_image = Image.open(io.BytesIO(image_data))
                processed_data[column_name].append(pil_image)
            else:
                # Directly append non-image data
                processed_data[column_name].append(example[column_name])

    return processed_data





def tokenize_captions(captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length, 
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    return inputs.input_ids[0] # NOTE: We want to shard (N,77) and not  (N,1,77) where N is the batch size

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1) # HWC to CHW as the VAE model expects CHW format

def f(x):
    if isinstance(x,str):
        return tokenize_captions(x) 

    image = convert_to_np(x, 256)
    image = 2 * (image / 255) - 1
    return image

def create_tf_dataset(data):
    # Convert data to TensorFlow Dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    # tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)  # Drop the last incomplete batch
    return tf_dataset


# Tensorflow dataset from a local Parquet file
def get_tf_dataset_from_parquet(parquet_file='data/train-00000-of-00262-57cebf95b4a9170c.parquet'):

    xs = read_dataset_from_parquet(parquet_file)
    data = jax.tree.map(f, xs)

    # Define the column names per dataset
    _column_names = timbrooks_instructpix2pix_clip_filtered_col_names = ["original_image", "edit_prompt", "edited_image"]

    # NOTE: tf.data.Dataset.from_tensor_slices divides up the dataset along the first or batch dimension
    tf_dataset = create_tf_dataset({
        "original_pixel_values": np.array(data[_column_names[0]]),
        "edited_pixel_values": np.array(data[_column_names[2]]),
        "input_ids": np.array(data[_column_names[1]])
    })
    return tf_dataset

# Tensorflow dataset from a Hugging Face dataset 
def get_tf_dataset_from_hf(dataset_path_or_name='fusing/instructpix2pix-1000-samples', split='train'):
    xs = read_dataset_from_arrow(dataset_path_or_name, split)
    data = jax.tree.map(f, xs)

    # Define the column names per dataset
    _column_names = fusing_instructpix2pix_1000_samples_col_names = ["input_image", "edit_prompt", "edited_image"]

    # NOTE: tf.data.Dataset.from_tensor_slices divides up the dataset along the first or batch dimension
    tf_dataset = create_tf_dataset({
        "original_pixel_values": np.array(data[_column_names[0]]),
        "edited_pixel_values": np.array(data[_column_names[2]]),
        "input_ids": np.array(data[_column_names[1]])
    })
    return tf_dataset


# Displaying and saving TF dataset images

def to_numpy(original, edited, ids):
    # print(original.shape, edited.shape)
    # original_max = tf.reduce_max(original)
    # original_min = tf.reduce_min(original)
    # edited_max = tf.reduce_max(edited)
    # edited_min = tf.reduce_min(edited)

    # Normalize images from [-1, 1] to [0, 255] using TensorFlow
    original = ((original + 1) * 127.5)
    edited = ((edited + 1) * 127.5)

    # Ensure casting to uint8 within the TensorFlow graph
    original = tf.cast(original, tf.uint8)
    edited = tf.cast(edited, tf.uint8)

    # Transpose images if they are in CHW format to HWC
    if tf.shape(original)[0] == 3:
        original = tf.transpose(original, [1, 2, 0])
    if tf.shape(edited)[0] == 3:
        edited = tf.transpose(edited, [1, 2, 0])

    # Convert tensors to numpy arrays
    original_np = original.numpy()
    edited_np = edited.numpy()
    ids_np = ids.numpy()

    # Convert numpy arrays to PIL Images
    original_pil = Image.fromarray(original_np)
    edited_pil = Image.fromarray(edited_np)

    return original_pil, edited_pil, ids_np

def get_images(tf_dataset, tokenizer, N=4):
    # Take N samples from the dataset
    xs = tf_dataset.take(N)

    # Convert dataset to numpy arrays and handle images
    numpy_dataset = xs.map(lambda x: tf.py_function(
        to_numpy, 
        [x['original_pixel_values'], x['edited_pixel_values'], x['input_ids']], 
        [tf.uint8, tf.uint8, tf.int64]))

    # Collect data
    batch = list(numpy_dataset.as_numpy_iterator())

    # Decode prompts
    input_ids_list = [i[2] for i in batch]
    prompt_list = tokenizer.batch_decode(input_ids_list, skip_special_tokens=True)

    return batch, prompt_list

def plot_images(tf_dataset, tokenizer, N=4):
    batch, prompt_list = get_images(tf_dataset, tokenizer, N)
    all_prompts = []
    # Setup the plot
    fig, axs = plt.subplots(2, N, figsize=(N*4, 8))
    for i, (original_image, edited_image, _) in enumerate(batch):
        axs[0, i].imshow(original_image)
        axs[1, i].imshow(edited_image)
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Prompt: {prompt_list[i]}", fontsize=10)
        all_prompts.append((original_image, edited_image, prompt_list[i]))

    plt.show()

def save_images(tf_dataset, tokenizer, N=4):
    batch, prompt_list = get_images(tf_dataset, tokenizer, N)
    all_prompts = [[orig, ed, prompt_list[i]] for i, (orig, ed, prompt) in enumerate(batch)]

    # Save images and prompts in the given directory
    RESOLUTION = 512
    SAVE_DIR = "lazy_eval_images"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    # Save images and prompts in the given directory
    for i, (original_image, edited_image, prompt) in enumerate(all_prompts):
        # images are numpy arrays so to save them  as images we need to convert them to PIL images
        original_image = Image.fromarray(original_image).resize((RESOLUTION, RESOLUTION))
        edited_image = Image.fromarray(edited_image).resize((RESOLUTION, RESOLUTION))
        original_image.save(f"{SAVE_DIR}/original_{i}.png")
        edited_image.save(f"{SAVE_DIR}/edited_{i}.png")
        with open(f"{SAVE_DIR}/prompt_{i}.txt", "w") as f:
            f.write(prompt)
#
#  %%


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", dtype=jnp.bfloat16) 
# tf_dataset = get_tf_dataset_from_parquet()
# plot_images(tf_dataset, tokenizer, N=5)
# save_images(tf_dataset, tokenizer, N=20)

# %%


"""
# Methods available on the tf.data.Dataset object:

$ dir(tf_dataset) # '_TensorSliceDataset'

['_GeneratorState',
'__abstractmethods__',
'__bool__',
'__class__',
'__class_getitem__',
'__debug_string__',
'__delattr__',
'__dict__',
'__dir__',
'__doc__',
'__eq__',
'__format__',
'__ge__',
'__getattribute__',
'__gt__',
'__hash__',
'__init__',
'__init_subclass__',
'__iter__',
'__le__',
'__len__',
'__lt__',
'__module__',
'__ne__',
'__new__',
'__nonzero__',
'__reduce__',
'__reduce_ex__',
'__repr__',
'__setattr__',
'__sizeof__',
'__slots__',
'__str__',
'__subclasshook__',
'__tf_tracing_type__',
'__weakref__',
'_abc_impl',
'_add_trackable_child',
'_add_variable_with_custom_getter',
'_apply_debug_options',
'_as_serialized_graph',
'_checkpoint_dependencies',
'_common_args',
'_consumers',
'_convert_variables_to_tensors',
'_copy_trackable_to_cpu',
'_deferred_dependencies',
'_deserialization_dependencies',
'_deserialize_from_proto',
'_export_to_saved_model_graph',
'_flat_shapes',
'_flat_structure',
'_flat_types',
'_functions',
'_gather_saveables_for_checkpoint',
'_graph',
'_graph_attr',
'_handle_deferred_dependencies',
'_inputs',
'_lookup_dependency',
'_maybe_initialize_trackable',
'_maybe_track_assets',
'_metadata',
'_name',
'_name_based_attribute_restore',
'_name_based_restores',
'_no_dependency',
'_object_identifier',
'_options',
'_options_attr',
'_options_tensor_to_options',
'_preload_simple_restoration',
'_restore_from_tensors',
'_serialize_to_proto',
'_serialize_to_tensors',
'_setattr_tracking',
'_shape_invariant_to_type_spec',
'_structure',
'_tensors',
'_tf_api_names',
'_tf_api_names_v1',
'_trace_variant_creation',
'_track_trackable',
'_trackable_children',
'_type_spec',
'_unconditional_checkpoint_dependencies',
'_unconditional_dependency_names',
'_update_uid',
'_variant_tensor',
'_variant_tensor_attr',
'apply',
'as_numpy_iterator',
'batch',
'bucket_by_sequence_length',
'cache',
'cardinality',
'choose_from_datasets',
'concatenate',
'counter',
'element_spec',
'enumerate',
'filter',
'flat_map',
'from_generator',
'from_tensor_slices',
'from_tensors',
'get_single_element',
'group_by_window',
'ignore_errors',
'interleave',
'list_files',
'load',
'map',
'options',
'padded_batch',
'prefetch',
'ragged_batch',
'random',
'range',
'rebatch',
'reduce',
'rejection_resample',
'repeat',
'sample_from_datasets',
'save',
'scan',
'shard',
'shuffle',
'skip',
'snapshot',
'sparse_batch',
'take',
'take_while',
'unbatch',
'unique',
'window',
'with_options',
'zip'] 
"""