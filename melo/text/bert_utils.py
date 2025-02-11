import torch
from transformers import AutoTokenizer, AutoModel

def get_bert_feature(text, token_idx_list, device):
    """Get BERT features for the text"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state[0]
        
    features = []
    for idx in token_idx_list:
        if idx < last_hidden_states.shape[0]:
            features.append(last_hidden_states[idx])
        else:
            features.append(torch.zeros_like(last_hidden_states[0]))
            
    return torch.stack(features)

def get_bert(text, word2ph, language, device):
    """Get BERT features for text with word-to-phoneme mapping"""
    if language == "ZH":
        bert_features = get_bert_feature(text, list(range(len(text))), device)
    else:
        words = text.split(" ")
        bert_features = get_bert_feature(text, list(range(len(words))), device)
    
    bert = []
    for i, w2p in enumerate(word2ph):
        for j in range(w2p):
            bert.append(bert_features[i])
    bert = torch.stack(bert, dim=0).transpose(0, 1)
    
    return bert
