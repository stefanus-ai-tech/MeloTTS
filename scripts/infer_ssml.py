import torch
from pathlib import Path
import torchaudio
from melo.models import MeloTTS

def main():
    # Setup paths
    checkpoint_path = Path("/home/adri/Documents/GitHub/MeloTTS/melo/checkpoints/checkpoint.pth")
    ssml_path = Path("/home/adri/Documents/GitHub/MeloTTS/test_ssml.txt")
    output_path = Path("output_ssml.wav")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeloTTS.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    # Read SSML content
    with open(ssml_path, 'r') as f:
        ssml_text = f.read()

    # Generate audio
    with torch.no_grad():
        audio = model.tts_with_ssml(ssml_text)
    
    # Ensure audio is in float32 format
    audio = audio.float()
    
    # Save the audio
    torchaudio.save(output_path, audio.cpu(), sample_rate=model.sample_rate)
    print(f"Generated audio saved to: {output_path}")

if __name__ == "__main__":
    main()
