import re
from unidecode import unidecode

def convert_to_ascii(text):
    """Convert non-ASCII characters to closest ASCII equivalent"""
    return unidecode(text)

def lowercase(text):
    """Convert text to lowercase"""
    return text.lower()

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

def collapse_whitespace(text):
    """Collapse multiple whitespace characters into single space"""
    return re.sub(r'\s+', ' ', text).strip()

def remove_aux_symbols(text):
    """Remove auxiliary symbols from text"""
    text = re.sub(r'[\<\>\(\)\[\]\"]+', '', text)
    return text

def english_cleaners2(text):
    """Pipeline for English text"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    text = remove_aux_symbols(text)
    return text
