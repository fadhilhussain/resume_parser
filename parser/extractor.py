import re 
import pandas as pd
import spacy
from fuzzywuzzy import process

nlp = spacy.load('en_core_web_sm')

#database of skills and educations
df = pd.read_csv('skills.csv')

def extract_data(text):
    return {
        'name' : extract_name(text),
        'email' : extract_email(text),
        'phone' : extract_phone(text),
        'place' : extract_place(text),
        'education' : extract_education(text),
        'skills' : extract_skill(text),
        'experience' : extract_experience(text)
    }

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_email(text):
    match = re.search(r'\S+@\S+',text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r'\+?\d[\d\s-]{8,}\d',text)
    return match.group(0) if match else None

def extract_place(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_=="GPE"))

def extract_education(text):
    text = text.lower()
    return list(set([edu for edu in df['education'] if edu in text]))

def extract_skill(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    skills = process.extractBests(" ".join(tokens), df['skill'], score_cutoff=80) #must refer ['python' and 'pythn' is 95% same ]
    return list(set([s[0] for s in skills]))

def extract_experience(text):
    match = re.findall(r'(\d+)\+?\s+(years|yrs)\s+experience',text.lower())
    return match[0][0] if match else None