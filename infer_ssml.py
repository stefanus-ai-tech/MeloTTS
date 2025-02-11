import argparse
import torch
from pathlib import Path
from melo.models import MeloTTS
import logging
import torchaudio
import re
from xml.etree import ElementTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    text = re.sub(r'[^\w\s.,?!;:\'\"-]', '', text)
    text = text.strip()
    return text

def ssml_to_text(ssml_text):
    try:
        root = ElementTree.fromstring(ssml_text)
        text_parts = []
        for element in root.itertext():
            text = clean_text(element)
            if text:
                text_parts.append(text)
        return ' '.join(text_parts)
    except Exception as e:
        logger.error(f"Error parsing SSML: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssml", type=str, required=True, help="Path to SSML input file")
    parser.add_argument("--output", type=str, required=True, help="Path to output WAV file")
    parser.add_argument("--checkpoint", type=str, default="melo/checkpoints/checkpoint.pth", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = MeloTTS.from_pretrained(Path(args.checkpoint))
        model = model.to(args.device)
        model.eval()

        # Read SSML file
        logger.info(f"Reading SSML from {args.ssml}")
        with open(args.ssml, 'r', encoding='utf-8') as f:
            ssml_text = f.read()

        # Extract text from SSML
        text = ssml_to_text(ssml_text)
        if not text:
            logger.error("No text extracted from SSML. Aborting.")
            return
        
        # Generate audio
        logger.info("Generating audio...")
        with torch.no_grad():
            audio = model.tts_with_ssml(ssml_text)
            
            # Move audio to CPU and ensure correct format
            audio = audio.cpu()
            
            # Ensure audio is 2D [channels, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add channel dimension
            elif audio.dim() > 2:
                audio = audio.squeeze()  # Remove any extra dimensions
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # Add channel dimension

            # Normalize audio
            audio = audio / torch.max(torch.abs(audio))

            # Ensure we have the correct shape for torchaudio.save()
            if audio.dim() != 2:
                raise ValueError(f"Expected 2D tensor [channels, samples], got shape {audio.shape}")

        # Save output
        logger.info(f"Saving audio to {args.output}")
        sample_rate = getattr(model, 'sample_rate', 22050)  # Get sample rate or use default
        torchaudio.save(args.output, audio.unsqueeze(0) if audio.dim() == 1 else audio, sample_rate)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()
