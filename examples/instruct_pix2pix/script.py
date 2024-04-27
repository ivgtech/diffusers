import jax
import requests

def get_parquet_files_list(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Assumes this returns a list of URLs directly
    else:
        raise Exception(f"Failed to retrieve Parquet files. Status code: {response.status_code}")

def main():
    # URL to fetch the parquet files list
    url = "https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train"

    # Fetch the list of files
    parquet_files = get_parquet_files_list(url)

    # Get the current JAX process device index
    worker_id = jax.process_index()
    num_workers = jax.process_count()

    # Calculate which files this worker should download
    total_files = len(parquet_files)
    files_per_worker = total_files // num_workers
    start_index = worker_id * files_per_worker
    end_index = start_index + files_per_worker

    if worker_id == num_workers - 1:  # Ensure the last worker picks up any remaining files
        end_index = total_files

    # Select the slice of files for this worker
    worker_files = parquet_files[start_index:end_index]

    # Write the list of files this worker should download to a file
    with open(f"worker_{worker_id}_files.txt", "w") as file:
        for url in worker_files:
            file.write(f"{url}\n")

    print(f"SCRIPT.PY: {end_index - start_index} URLs assigned for download by worker {worker_id}.")


if __name__ == "__main__":
    main()

