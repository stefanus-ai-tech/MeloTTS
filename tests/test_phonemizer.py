import unittest
from melo.text.phonemizer import Phonemizer

class TestPhonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = Phonemizer()

    def test_english_phonemization(self):
        text = "Hello world"
        phonemes = self.phonemizer.phonemize(text)
        self.assertTrue(len(phonemes) > 0)
        self.assertIn('əʊ', phonemes)  # Should contain 'o' sound
