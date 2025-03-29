# Function to clean and tokenize text
def clean_text(text):
    # Remove non-alphabetic characters and split into words
    words = re.findall(r'\b\w+\b', str(text).lower())
    return words

# Function to clean text, remove stopwords, and return processed text
def preprocess_text(text):
    words = clean_text(text)  # Tokenize and remove non-alphabetic characters
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)  # Reconstruct sentence

import re 

# Function to clean and tokenize text
def clean_text(text):
    # Remove non-alphabetic characters and split into words
    words = re.findall(r'\b\w+\b', str(text).lower())
    return words