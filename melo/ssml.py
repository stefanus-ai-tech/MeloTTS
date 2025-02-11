from lxml import etree

def _handle_speak_tag(element, ssml_attributes):
    """Handles the <speak> tag."""
    if element.tag != 'speak':
        raise ValueError("Invalid SSML: Root element must be <speak>")
    # Handle attributes like version, xmlns, xml:lang if needed
    return _process_children(element, ssml_attributes)

def _handle_voice_tag(element, ssml_attributes):
    """Handles the <voice> tag."""
    voice_name = element.get('name')
    ssml_attributes['voice_name'] = voice_name
    return _process_children(element, ssml_attributes)

def _handle_break_tag(element, ssml_attributes):
    """Handles the <break> tag."""
    time = element.get('time')
    strength = element.get('strength')
    ssml_attributes.setdefault('break', []).append({'time': time, 'strength': strength})
    return ""

def _handle_sub_tag(element, ssml_attributes):
    """Handles the <sub> tag."""
    return element.get('alias', '')

def _handle_say_as_tag(element, ssml_attributes):
    """Handles the <say-as> tag."""
    interpret_as = element.get('interpret-as')
    text = _process_children(element, ssml_attributes)  # Get the text content of the tag

    if interpret_as == 'characters':
        return ' '.join(list(text))
    elif interpret_as == 'spell-out':
        return ' '.join(list(text))
    elif interpret_as == 'telephone':
        # In a real implementation, we'd format this properly
        return text
    elif interpret_as == 'date':
        # In a real implementation, we'd format the date based on the 'format' attribute
        return text
    elif interpret_as == 'cardinal':
        return text
    elif interpret_as == 'ordinal':
        return text
    elif interpret_as == 'fraction':
        return text
    elif interpret_as == 'measure':
        return text
    elif interpret_as == 'time':
        return text
    elif interpret_as == 'address':
        return text
    else:
        return text

def _handle_emphasis_tag(element, ssml_attributes):
    """Handles the <emphasis> tag."""
    # level = element.get('level') # Get the emphasis level
    return _process_children(element, ssml_attributes)

def _handle_mstts_express_as_tag(element, ssml_attributes):
    """Handles the <mstts:express-as> tag."""
    # style = element.get('style')
    return _process_children(element, ssml_attributes)

def _handle_prosody_tag(element, ssml_attributes):
    """Handles the <prosody> tag."""
    rate = element.get('rate')
    pitch = element.get('pitch')
    volume = element.get('volume')
    # contour = element.get('contour') # Complex attribute, handle later
    # print(f"Prosody: rate={rate}, pitch={pitch}, volume={volume}") # Debug print
    return _process_children(element, ssml_attributes)

def _handle_phoneme_tag(element, ssml_attributes):
    """Handles the <phoneme> tag."""
    alphabet = element.get('alphabet')
    ph = element.get('ph')
    # print(f"Phoneme: alphabet={alphabet}, ph={ph}") # Debug print
    return ph

def _handle_audio_tag(element, ssml_attributes):
    """Handles the <audio> tag."""
    src = element.get('src')
    # print(f"Audio src: {src}") # Debug print
    return "" # Return empty string for now
    
def _handle_p_tag(element, ssml_attributes):
    """Handles the <p> tag."""
    # Paragraph tags, for now treat as containers
    return _process_children(element, ssml_attributes) + " "

def _process_children(element, ssml_attributes):
    """Recursively processes the children of an element."""
    result = []
    for child in element:
        if child.tag == 'voice':
            result.append(_handle_voice_tag(child, ssml_attributes))
        elif child.tag == 'break':
            result.append(_handle_break_tag(child, ssml_attributes))
        elif child.tag == 'p':
            result.append(_handle_p_tag(child, ssml_attributes))
        elif child.tag == 'sub':
            result.append(_handle_sub_tag(child, ssml_attributes))
        elif child.tag == 'say-as':
            result.append(_handle_say_as_tag(child, ssml_attributes))
        elif child.tag == 'emphasis':
            result.append(_handle_emphasis_tag(child, ssml_attributes))
        elif child.tag == '{https://www.w3.org/2001/mstts}express-as':
            result.append(_handle_mstts_express_as_tag(child, ssml_attributes))
        elif child.tag == 'prosody':
            result.append(_handle_prosody_tag(child, ssml_attributes))
        elif child.tag == 'phoneme':
            result.append(_handle_phoneme_tag(child, ssml_attributes))
        elif child.tag == 'audio':
            result.append(_handle_audio_tag(child, ssml_attributes))
        elif child.text:
            result.append(child.text)
        if child.tail:  # Handle text that follows the tag
            result.append(child.tail)
    return ''.join(result)

def extract_text_from_ssml(ssml_string):
    """
    Extracts text content from an SSML string, handling various tags.
    """
    try:
        root = etree.fromstring(ssml_string)
        ssml_attributes = {} # create dict
        return _handle_speak_tag(root, ssml_attributes)
    except etree.XMLSyntaxError:
        return ssml_string
