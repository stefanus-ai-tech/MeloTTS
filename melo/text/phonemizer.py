import re
from phonemizer import phonemize
from phonemizer.separator import Separator

class Phonemizer:
    def __init__(self):
        self.separator = Separator(word=' ', syllable='|', phone='')

    def phonemize_english(self, text: str) -> str:
        """Convert English text to phonemes using espeak"""
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        phonemes = phonemize(
            text,
            backend='espeak',
            language='en-us',
            preserve_punctuation=True,
            strip=True,
            with_stress=True,
            separator=self.separator
        )
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        return phonemes

    def phonemize(self, text: str, language: str = 'en') -> str:
        if language.lower() in ['en', 'en-us']:
            return self.phonemize_english(text)
        return text
