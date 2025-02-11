import os
import json
import torch
import numpy as np
import soundfile as sf
from melo import commons
from melo import utils
from melo.models import SynthesizerTrn
from melo.text import text_to_sequence, _clean_text, language_id_map
from torch import no_grad, LongTensor
import logging
from melo.ssml import extract_text_from_ssml

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_imports():
    """Debug function to check module paths and imports"""
    import sys
    logger.debug("Python path:")
    for path in sys.path:
        logger.debug(f"  {path}")
    
    logger.debug("Checking melo package structure:")
    melo_path = os.path.dirname(commons.__file__)
    logger.debug(f"Melo package path: {melo_path}")
    
    # List contents of melo directory
    for root, dirs, files in os.walk(melo_path):
        logger.debug(f"\nDirectory: {root}")
        for d in dirs:
            logger.debug(f"  Dir: {d}")
        for f in files:
            logger.debug(f"  File: {f}")

def get_text(text, language, tone=0):
    """Convert text to sequence with language handling"""
    if isinstance(language, str):
        language = language_id_map.get(language, 0)
    text_norm = text_to_sequence(text)
    if tone is None:
        tone = 0
    tone = LongTensor([tone])
    language = LongTensor([language])
    return text_norm, tone, language

def ex_print(text, escape=False):
    if escape:
        print(text.replace("\n", "\\n").replace("\t", "\\t"))
    else:
        print(text)

def infer_from_ssml(ssml_file_path, output_path, model_path="outputs/EN-US"):
    try:
        logger.info(f"Loading config from {model_path}")
        config_path = os.path.join(model_path, "config.json") 
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
        
        logger.debug("Initializing model...")
        hps = utils.HParams(**config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            spk2id=hps.data.spk2id,
            sampling_rate=hps.data.sampling_rate,
            **hps.model,
        ).to(device)
        
        _ = net_g.eval()

        # Load checkpoint
        ckpt = torch.load(f"{model_path}/G_100000.pth", map_location=device)
        net_g.load_state_dict(ckpt["model"])

        # Read SSML file
        with open(ssml_file_path, "r") as f:
            ssml_text = f.read()

        # Process SSML
        ssml_data = extract_text_from_ssml(ssml_text)
        text = ssml_data["text"]
        ssml_attributes = ssml_data["ssml_attributes"]

        # Clean and convert text
        text = _clean_text(text, ["english_cleaners2"])
        stn_tst, tone, language = get_text(text, 0, tone=0)
        
        with no_grad():
            x_tst = torch.LongTensor(stn_tst).unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.shape[0]]).to(device)
            tone = tone.to(device)
            language = language.to(device)
            
            # Use speaker ID for EN-US voice
            sid = LongTensor([hps.data.spk2id["EN-US"]]).to(device)

            # Generate audio
            audio = net_g.infer(
                x=x_tst, 
                x_lengths=x_tst_lengths,
                sid=sid,
                tone=tone,
                language=language,
                noise_scale=.667,
                noise_scale_w=0.8,
                length_scale=1,
                bert=None,
                ja_bert=None,
                ssml_attributes=ssml_attributes
            )[0][0, 0].data.cpu().float().numpy()

            # Save audio
            sf.write(output_path, audio, samplerate=hps.data.sampling_rate)
            logging.info(f"Audio saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting MeloTTS inference")
        debug_imports()
        
        if not os.path.exists("test_ssml.txt"):
            logger.error("test_ssml.txt not found")
            exit(1)
            
        infer_from_ssml(
            "test_ssml.txt",
            "output.wav"
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
