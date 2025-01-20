import json
from collections import defaultdict

def create_inverted_index(input_json, output_json):
    """
    1. Διαβάζει τα καθαρισμένα δεδομένα από το `input_json`.
    2. Δημιουργεί ένα inverted index με (λέξη -> λίστα doc_ids).
    3. Αποθηκεύει το inverted index στο αρχείο `output_json`.
    """
    # Φορτώνουμε το cleaned JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Το 'inverted_index' θα είναι dictionary: λέξη -> σε ποια docs εμφανίζεται
    # Χρησιμοποιούμε defaultdict(list) για εύκολο append
    inverted_index = defaultdict(list)

    # Διατρέχουμε όλα τα έγγραφα
    # Θα χρησιμοποιήσουμε το i ως doc_id (i+1 αν θέλεις να ξεκινά από το 1)
    for i, doc in enumerate(data):
        doc_id = i + 1  # Έτσι το πρώτο έγγραφο έχει doc_id=1
        tokens = doc.get('cleaned_definition', [])

        # Για κάθε λέξη, προσθέτουμε το doc_id στη λίστα
        for token in tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)

    # Μετατρέπουμε το defaultdict σε κανονικό dict
    inverted_index = dict(inverted_index)

    # Αποθηκεύουμε σε αρχείο JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)

    print(f"Inverted index created and saved to {output_json}")


if __name__ == "__main__":
    input_file = "wiki_definitions_cleaned.json"  # το αρχείο που φτιάξαμε μετά το stemming/stopword removal
    output_file = "inverted_index.json"
    
    create_inverted_index(input_file, output_file)