import json
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    """
    Δέχεται ένα κείμενο (string) και επιστρέφει μια λίστα από tokens που
    έχουν υποστεί: 
      - lowercase
      - αφαίρεση ειδικών χαρακτήρων
      - tokenization
      - stopword removal
      - stemming
    """

    # 1. Μετατροπή σε πεζά
    text = text.lower()

    # 2. Αφαίρεση ειδικών χαρακτήρων (κρατάμε μόνο γράμματα, αριθμούς, κενά)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3. Tokenization
    tokens = text.split()

    # 4. Αφαίρεση stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # 5. Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return stemmed_tokens

def main(input_json, output_json):
    """
    1. Διαβάζει το JSON με τα αρχικά δεδομένα (input_json).
    2. Κάνει preprocess το πεδίο 'definition' σε κάθε εγγραφή.
    3. Αποθηκεύει τα νέα δεδομένα σε άλλο JSON (output_json).
    """

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)


    cleaned_data = []

    for item in data:

        original_definition = item.get('definition', '')


        cleaned_tokens = preprocess_text(original_definition)

        new_item = {
            'url': item.get('url', ''),
            'title': item.get('title', ''),
            'original_definition': original_definition,
            'cleaned_definition': cleaned_tokens  # λίστα με tokens μετά το stemming κλπ.
        }

        cleaned_data.append(new_item)


    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"Preprocessing complete. Saved to {output_json}")

if __name__ == "__main__":
    output_file = "wiki_definitions_cleaned.json"

    main(input_file, output_file)
