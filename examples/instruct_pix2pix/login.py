import yaml
from huggingface_hub import HfFolder, notebook_login, login
import wandb


def load_secrets(file_path):
    """Load API keys from a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def login_to_wandb(api_key):
    """Login to Weights & Biases."""
    wandb.login(key=api_key)


def login_to_huggingface(token):
    """Login to Hugging Face."""
    HfFolder.save_token(
        token
    )  # This saves the token for use by the transformers library
    notebook_login(token)


def main():
    # Load secrets from YAML file
    secrets = load_secrets("secrets.yaml")

    login_to_wandb(secrets["wandb"]["token"])
    login(token=secrets["huggingface"]["token"])

    print("Logged into both wandb and Hugging Face successfully.")


if __name__ == "__main__":
    main()
