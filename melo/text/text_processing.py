import re
import logging

logger = logging.getLogger(__name__)

# Basic character set
_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate([_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa))}
_id_to_symbol = {i: s for i, s in enumerate([_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa))}

def text_to_sequence(text):
    """Converts text to a sequence of IDs using the above mappings"""
    sequence = []
    
    for symbol in text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
        else:
            logger.warning(f'Symbol {symbol} not in symbol set, skipping')
            continue
    
    return sequence

def sequence_to_text(sequence):
    """Converts a sequence of IDs back to text"""
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]
    return result

def cleaned_text_to_sequence(cleaned_text, tones, language_str, symbol_to_id=None):
    """Convert cleaned text to sequence with language and tone information"""
    from melo.text.symbols import language_id_map, language_tone_start_map
    
    if symbol_to_id is None:
        symbol_to_id = _symbol_to_id
        
    sequence = text_to_sequence(cleaned_text)
    
    if isinstance(language_str, str):
        language_id = language_id_map[language_str]
    else:
        language_id = language_str
        
    tone_start = language_tone_start_map.get(language_str, 0)
    tones = [t + tone_start for t in tones]
    
    return sequence, tones, language_id
