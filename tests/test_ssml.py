import unittest
from melo.text.ssml_parser import SSMLParser

class TestSSMLParser(unittest.TestCase):
    def setUp(self):
        self.parser = SSMLParser()
        
    def test_complex_ssml(self):
        ssml = """
        <speak>
            <prosody rate="slow">
                Test text
                <break time="500ms"/>
                <emphasis level="strong">Important!</emphasis>
            </prosody>
        </speak>
        """
        text, attributes = self.parser.parse(ssml)
        self.assertIn('prosody', attributes)
        self.assertIn('break', attributes)
        self.assertIn('emphasis', attributes)
