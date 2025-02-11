import torch
from pathlib import Path
from melo.models import MeloTTS

def test_basic_tts():
    # Load model
    model = MeloTTS.from_pretrained(Path("melo/checkpoints/checkpoint.pth"))
    model.eval()
    
    # Create simple SSML
    ssml = """<?xml version="1.0"?>
    <speak version="1.0">
        <voice name="en-US-JennyNeural">
            This is a test of the text to speech system.
        </voice>
    </speak>
    """
    
    # Generate audio
    with torch.no_grad():
        audio = model.tts_with_ssml(ssml)
    
    # Save output
    import torchaudio
    torchaudio.save("test_output.wav", audio, model.sample_rate)

if __name__ == "__main__":
    test_basic_tts()
