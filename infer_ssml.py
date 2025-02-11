import argparse
import torch
from pathlib import Path
from melo.models import MeloTTS
import logging
import torchaudio
import re
from xml.etree import ElementTree
import torch.nn.functional as F
from bs4 import BeautifulSoup
from typing import Dict, Any

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

def apply_compression(audio, threshold=0.8, ratio=4.0):
    """Apply dynamic range compression"""
    magnitude = torch.abs(audio)
    mask = magnitude > threshold
    compressed = audio.clone()
    compressed[mask] = threshold + (magnitude[mask] - threshold) / ratio
    return compressed

def enhance_audio(audio, sr=22050, iterations=2):
    """Recursively enhance audio quality"""
    enhanced = audio
    for i in range(iterations):
        # Apply smoothing
        enhanced = F.pad(enhanced, (1, 1), mode='replicate')
        enhanced = F.avg_pool1d(enhanced.unsqueeze(0), kernel_size=3, stride=1).squeeze(0)
        
        # Compress dynamic range
        enhanced = apply_compression(enhanced)
        
        # Normalize
        enhanced = enhanced / torch.max(torch.abs(enhanced))
        
    return enhanced

def parse_ssml(ssml_text: str) -> tuple[str, Dict[str, Any]]:
    """Parse SSML and extract text and attributes"""
    soup = BeautifulSoup(ssml_text, 'xml')
    
    # Extract attributes from SSML tags
    ssml_attributes = {
        'voice': [],
        'break': [],
        'prosody': [],
        'emphasis': [],
        'say-as': [],
        'sub': [],
        'p': [],
    }
    
    # Parse voice tags
    for voice in soup.find_all('voice'):
        ssml_attributes['voice'].append({
            'name': voice.get('name', ''),
            'text': voice.get_text().strip()
        })
    
    # Parse prosody tags
    for prosody in soup.find_all('prosody'):
        ssml_attributes['prosody'].append({
            'rate': prosody.get('rate', 'medium'),
            'pitch': prosody.get('pitch', '0st'),
            'volume': prosody.get('volume', '0dB'),
            'text': prosody.get_text().strip()
        })
        
    # Parse break tags
    for break_tag in soup.find_all('break'):
        ssml_attributes['break'].append({
            'time': break_tag.get('time', ''),
            'strength': break_tag.get('strength', '')
        })
        
    # Parse emphasis tags
    for emphasis in soup.find_all('emphasis'):
        ssml_attributes['emphasis'].append({
            'level': emphasis.get('level', 'moderate'),
            'text': emphasis.get_text().strip()
        })

    # Get plain text without XML
    text = soup.get_text().strip()
    return text, ssml_attributes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssml", type=str, required=True, help="Path to SSML input file")
    parser.add_argument("--output", type=str, required=True, help="Path to output WAV file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = MeloTTS.from_pretrained(Path(args.checkpoint))
        model = model.to(args.device)
        model.eval()

        # Read and parse SSML
        logger.info(f"Reading SSML from {args.ssml}")
        with open(args.ssml, 'r', encoding='utf-8') as f:
            ssml_text = f.read()
            
        # Parse SSML and extract features
        text, ssml_attributes = parse_ssml(ssml_text)
        if not text:
            logger.error("No text extracted from SSML. Aborting.")
            return

        # Generate audio
        logger.info("Generating audio...")
        with torch.no_grad():
            # Process SSML directly through model's tts_with_ssml method
            audio = model.tts_with_ssml(ssml_text)
            
            # Ensure audio is in correct format
            audio = audio.cpu()
            
            if len(audio.shape) == 1:
                # Single channel audio - add channel dimension
                audio = audio.unsqueeze(0)
            elif len(audio.shape) > 2:
                # Too many dimensions - squeeze and ensure channel dim
                audio = audio.squeeze()
                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)
            
            # Normalize audio
            max_val = torch.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            # Final format check
            if audio.dim() != 2:
                raise ValueError(f"Expected 2D tensor [channels, samples], got shape {audio.shape}")

        # Save output
        logger.info(f"Saving audio to {args.output}")
        torchaudio.save(args.output, audio, model.sample_rate)
        logger.info("Done!")

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()
