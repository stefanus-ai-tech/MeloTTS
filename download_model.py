import os
import argparse
from huggingface_hub import hf_hub_download

def download_model(language):
    """Downloads the model and config files for the specified language."""

    repo_id = "myshell-ai/MeloTTS-English"

    output_dir = os.path.join("outputs", language)
    os.makedirs(output_dir, exist_ok=True)

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=output_dir)
        print(f"Config path: {config_path}")

        model_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pth", local_dir=output_dir)
        print(f"Model path: {model_path}")

    except Exception as e:
        print(f"Error downloading config/model for language {language}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MeloTTS model.")
    parser.add_argument("language", type=str, help="Language code (e.g., EN-US)")
    args = parser.parse_args()
    download_model(args.language)
