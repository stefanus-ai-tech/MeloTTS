from bs4 import BeautifulSoup
import re
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SSMLParser:
    def __init__(self):
        self.supported_tags = {
            'speak', 'voice', 'break', 'prosody', 'emphasis',
            'say-as', 'sub', 'p', 'audio'
        }
        
        self.emphasis_levels = {
            'strong': 1.5,  # Increase volume/pitch
            'moderate': 1.2,
            'reduced': 0.8,
            'none': 1.0
        }
        
        self.rate_values = {
            'x-slow': 0.5,
            'slow': 0.75,
            'medium': 1.0,
            'fast': 1.5,
            'x-fast': 2.0
        }

    def parse(self, ssml_text: str) -> Tuple[str, Dict[str, Any]]:
        """Parse SSML text and return cleaned text and attributes"""
        soup = BeautifulSoup(ssml_text, 'xml')
        
        # Initialize attribute collectors
        attributes = {
            'prosody': [],
            'break': [],
            'emphasis': [],
            'say_as': [],
            'sub': [],
            'audio': [],
            'voice': []
        }
        
        # Process each tag type
        self._process_breaks(soup, attributes)
        self._process_prosody(soup, attributes)
        self._process_emphasis(soup, attributes)
        self._process_say_as(soup, attributes)
        self._process_sub(soup, attributes)
        self._process_audio(soup, attributes)
        self._process_voice(soup, attributes)
        
        # Get clean text with preserved spacing
        text = self._extract_text(soup)
        
        return text, attributes

    def _process_breaks(self, soup, attributes):
        for break_tag in soup.find_all('break'):
            break_attr = {}
            if 'time' in break_tag.attrs:
                time_str = break_tag['time']
                time_ms = self._parse_time(time_str)
                break_attr['time'] = time_ms
            if 'strength' in break_tag.attrs:
                break_attr['strength'] = break_tag['strength']
            attributes['break'].append(break_attr)

    def _process_prosody(self, soup, attributes):
        for prosody in soup.find_all('prosody'):
            attr = {
                'text': prosody.get_text(),
                'rate': self._parse_rate(prosody.get('rate', 'medium')),
                'pitch': self._parse_pitch(prosody.get('pitch', '0st')),
                'volume': self._parse_volume(prosody.get('volume', '0dB')),
                'start_char': len(self._extract_text(soup.find_previous_siblings())),
                'end_char': len(self._extract_text(soup.find_previous_siblings())) + len(prosody.get_text())
            }
            attributes['prosody'].append(attr)

    def _parse_time(self, time_str: str) -> int:
        """Convert SSML time to milliseconds"""
        match = re.match(r'(\d+)(ms|s)', time_str)
        if not match:
            return 0
        value, unit = match.groups()
        if unit == 'ms':
            return int(value)
        return int(float(value) * 1000)

    def _parse_rate(self, rate: str) -> float:
        """Parse speech rate value"""
        if rate in self.rate_values:
            return self.rate_values[rate]
        if rate.endswith('%'):
            return float(rate.rstrip('%')) / 100
        return 1.0

    def _parse_pitch(self, pitch: str) -> float:
        """Parse pitch modification value"""
        if pitch.endswith('st'):  # Semitones
            return float(pitch.rstrip('st'))
        if pitch.endswith('%'):
            return float(pitch.rstrip('%')) / 100
        return 0.0

    def _parse_volume(self, volume: str) -> float:
        """Parse volume modification value"""
        if volume.endswith('dB'):
            return float(volume.rstrip('dB'))
        if volume.endswith('%'):
            return float(volume.rstrip('%')) / 100
        return 1.0

    def _extract_text(self, element) -> str:
        """Extract text while preserving whitespace"""
        return ' '.join(node.strip() for node in element.stripped_strings)
