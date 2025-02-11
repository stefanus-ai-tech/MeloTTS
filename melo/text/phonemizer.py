import re
from typing import List
import pypinyin
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

class Phonemizer:
    def __init__(self):
        self.backend = EspeakBackend(language='en-us')
        self.separator = Separator(word=' ', syllable='|', phone='')
        
    def phonemize_english(self, text: str) -> str:
        """Convert English text to phonemes using espeak"""
        # Clean text
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        # Phonemize with espeak
        phonemes = phonemize(
            text,
            backend='espeak',
            language='en-us',
            preserve_punctuation=True,
            strip=True,
            with_stress=True,
            separator=self.separator
        )
        
        # Clean up phonemes
        phonemes = re.sub(r'\s+', ' ', phonemes)
        phonemes = phonemes.strip()
        
        return phonemes

    def phonemize(self, text: str, language: str = 'en') -> str:
        """Convert text to phonemes based on language"""
        if language.lower() in ['en', 'en-us']:
            return self.phonemize_english(text)
        # Add other language support here
        return text
