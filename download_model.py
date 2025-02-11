from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(repo_id="myshell-ai/MeloTTS-English", filename="checkpoint.pth")
config_path = hf_hub_download(repo_id="myshell-ai/MeloTTS-English", filename="config.json")

print(f"Checkpoint path: {ckpt_path}")
print(f"Config path: {config_path}")
