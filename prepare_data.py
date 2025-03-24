import json
import pandas as pd
import spacy
import re

# Load intents
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# Load spaCy model with only needed components
nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])

def preprocess(text):
    # Lowercase and strip whitespace
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        # Skip punctuation and stop words
        if not token.is_alpha or token.is_stop:
            continue

        # Check if any child of the token is a negation
        negation = any(child.dep_ == "neg" for child in token.children)
        if negation:
            tokens.append("not_" + token.lemma_)
        else:
            tokens.append(token.lemma_)
            
    return " ".join(tokens)

data = []
for intent in intents:
    for pattern in intent['patterns']:
        cleaned_text = preprocess(pattern)
        data.append({'text': cleaned_text, 'intent': intent['tag']})

df = pd.DataFrame(data)
df.to_csv('intents_dataset.csv', index=False)
