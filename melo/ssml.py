from lxml import etree

def _handle_speak_tag(element, ssml_attributes):
    """Handles the <speak> tag."""
    if element.tag != 'speak':
        raise ValueError("Invalid SSML: Root element must be <speak>")
    ssml_attributes['version'] = element.get('version')
    ssml_attributes['lang'] = element.get('{http://www.w3.org/XML/1998/namespace}lang')
    return _process_children(element, ssml_attributes)

def _handle_voice_tag(element, ssml_attributes):
    """Handles the <voice> tag."""
    voice_name = element.get('name')
    ssml_attributes.setdefault('voice', []).append({'name': voice_name})
    return _process_children(element, ssml_attributes)

def _handle_break_tag(element, ssml_attributes):
    """Handles the <break> tag."""
    time = element.get('time')
    strength = element.get('strength')
    ssml_attributes.setdefault('break', []).append({'time': time, 'strength': strength, 'text': ""})
    return ""

def _handle_sub_tag(element, ssml_attributes):
    """Handles the <sub> tag."""
    alias = element.get('alias', '')
    ssml_attributes.setdefault('sub', []).append({'alias': alias, 'text': element.text})
    return alias

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
    
    ssml_attributes.setdefault('say-as', []).append({'interpret-as': interpret_as, 'text': text})
    return text

def _handle_emphasis_tag(element, ssml_attributes):
    """Handles the <emphasis> tag."""
    level = element.get('level') # Get the emphasis level
    ssml_attributes.setdefault('emphasis', []).append({'level': level, 'text': _process_children(element, ssml_attributes)})
    return _process_children(element, ssml_attributes)

def _handle_mstts_express_as_tag(element, ssml_attributes):
    """Handles the <mstts:express-as> tag."""
    style = element.get('style')
    ssml_attributes.setdefault('mstts:express-as', []).append({'style': style, 'text': _process_children(element, ssml_attributes)})
    return _process_children(element, ssml_attributes)

def _handle_prosody_tag(element, ssml_attributes):
    """Handles the <prosody> tag."""
    rate = element.get('rate')
    pitch = element.get('pitch')
    volume = element.get('volume')
    contour = element.get('contour') # Complex attribute, handle later
    ssml_attributes.setdefault('prosody', []).append({'rate': rate, 'pitch': pitch, 'volume': volume, 'contour': contour, 'text': _process_children(element, ssml_attributes)})
    return _process_children(element, ssml_attributes)

def _handle_phoneme_tag(element, ssml_attributes):
    """Handles the <phoneme> tag."""
    alphabet = element.get('alphabet')
    ph = element.get('ph')
    ssml_attributes.setdefault('phoneme', []).append({'alphabet': alphabet, 'ph': ph, 'text': ph})
    return ph

def _handle_audio_tag(element, ssml_attributes):
    """Handles the <audio> tag."""
    src = element.get('src')
    ssml_attributes.setdefault('audio', []).append({'src': src, 'text': ""})
    return "" # Return empty string for now
    
def _handle_p_tag(element, ssml_attributes):
    """Handles the <p> tag."""
    # Paragraph tags, for now treat as containers
    ssml_attributes.setdefault('p', []).append({'text': _process_children(element, ssml_attributes)})
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
        if child.text:
            result.append(child.text)
    
    return ''.join(result)

def extract_text_from_ssml(ssml_string):
    """
    Extracts text content from an SSML string, handling various tags.
    Returns a dictionary containing the extracted text and SSML attributes.
    """
    try:
        root = etree.fromstring(ssml_string)
        ssml_attributes = {}  # create dict
        text = _handle_speak_tag(root, ssml_attributes)
        # print(ssml_attributes)
        return {
            'text': text,
            'ssml_attributes': ssml_attributes
        }

    except etree.XMLSyntaxError:
        return {'text': ssml_string, 'ssml_attributes': {}}
