# Description:
# This script downloads and shards a large dataset across multiple machines, saving each shard to disk.

# Hugging Face dataset name:                'timbrooks/instructpix2pix-clip-filtered'
# Size of downloaded dataset files:          130 GB
# Size of the auto-converted Parquet files:  130 GB
# Number of rows:                           313,010
# total number of parquet files:             262
# approx size of each parquet file:          500 MB
# aprox num. of examples per parquet file:   1,195
# schema: {'original_prompt': 'string', 'original_image': 'PIL image', 'edit_prompt': 'string', 'edited_prompt': 'string', 'edited_image': 'PIL image'}

import jax
from jax import pmap
import numpy as np
import os
import sys
import argparse
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from functools import partial


def split_range_into_intervals(start, end, N):
    # start and end represent indices on an array of length = (end - start)
    # example: split_range_into_intervals(0, 60_000, 4)
    # [(0, 15000), (15000, 30000), (30000, 45000), (45000, 60000)]

    # Validate inputs
    if not all(isinstance(x, int) for x in [start, end, N]):
        raise ValueError("All inputs must be integers.")
    if start >= end:
        raise ValueError("Start must be less than end.")
    if N <= 0:
        raise ValueError("Number of intervals N must be positive.")

    # Calculate the size of each interval
    total_size = end - start
    if N > total_size:
        N = total_size
    
    interval_size = total_size // N
    extra = total_size % N

    intervals = []
    current_start = start

    # Create each interval, distributing the remainder along the first few intervals
    for i in range(N):
        current_end = current_start + interval_size + (1 if i < extra else 0)
        intervals.append([current_start, current_end])
        current_start = current_end
    
    return np.array(intervals)

def main(dataset_name, total_length, percentage):
    # Determine this machine's index and the total number of machines
    vm_id = jax.process_index()
    N = n_vms = jax.process_count()

    # Calculate the total number of examples to process based on the percentage
    total_to_process = int(total_length * (percentage / 100))

    # Calculate the portion of the dataset to download per machine
    per_device = total_to_process // n_vms

    start_index = per_device * vm_id
    end_index = start_index + per_device if vm_id != n_vms - 1 else start_index + per_device + (total_to_process % n_vms)

    # np array with shape:  N x (start,end)
    # intervals = split_range_into_intervals(start_index, end_index, num_devices)

    print(f'train[{start_index}:{end_index}]')

    ds = load_dataset(dataset_name, split=f'train[{start_index}:{end_index}]')
    dataset = split_dataset_by_node(ds, rank=int(vm_id), world_size=int(N))

    save_path = f"./{dataset_name}_part_{vm_id}_{start_index}_{end_index}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset.save_to_disk(save_path)

    print(f"\n[{vm_id}]: {dataset_name} part from {start_index} to {end_index} saved to:\n{save_path}\n")

if __name__ == "__main__":
    # Example usage:
    # gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=us-central2-b --worker=all --command="source venv/bin/activate && python $PYTHON_SCRIPT 'cifar10' 60000"

    ds = 'timbrooks/instructpix2pix-clip-filtered'
    tl = 313010

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument('dataset_name', type=str, default=ds, help='Name of the dataset')
    parser.add_argument('total_length', type=int, default=tl, help='Total length of the dataset')
    parser.add_argument('--percentage', type=float, default=100, help='Percentage of the dataset to use (default 100%)')

    main(args.dataset_name, args.total_length, args.percentage)
    

