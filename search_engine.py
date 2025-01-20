import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def load_inverted_index(index_file):
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the file {index_file}.")
        return None

def load_documents(docs_file):
    """
    Φορτώνει τη λίστα των εγγράφων (με paragraphs) από JSON αρχείο.
    """
    try:
        with open(docs_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the file {docs_file}.")
        return None

def preprocess_query(query):
    """
    Προεπεξεργάζεται το ερώτημα του χρήστη:
    """
    # 1. Μετατροπή σε πεζά
    query = query.lower()
    
    # 2. Αφαίρεση ειδικών χαρακτήρων
    query = re.sub(r'[^a-z0-9\s]', ' ', query)
    
    # 3. Tokenization
    tokens = query.split()
    
    # 4. Stop-word removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # 5. Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return stemmed_tokens

def process_boolean_query(tokens, inverted_index):
    """
    Επεξεργάζεται ένα ερώτημα ΜΟΝΟ με Boolean Operators (AND, OR, NOT).
    tokens: λίστα από όρους (π.χ. ["machin", "and", "learn"])
    
    Επιστρέφει ένα set() με τα IDs των εγγράφων που ταιριάζουν στο ερώτημα.
    """
    result_set = None
    operation = None

    for token in tokens:
        if token in ["and", "or", "not"]:
            operation = token
        else:
            # βρίσκουμε τα έγγραφα όπου εμφανίζεται
            docs_for_token = set(map(int, inverted_index.get(token, [])))

            if result_set is None:
                # πρώτη λέξη
                result_set = docs_for_token
            else:
                if operation == "and":
                    result_set &= docs_for_token
                elif operation == "or":
                    result_set |= docs_for_token
                elif operation == "not":
                    result_set -= docs_for_token
                else:
                    # fallback σε OR
                    result_set |= docs_for_token

    if result_set is None:
        result_set = set()

    return result_set

def process_simple_query(tokens, inverted_index, default_operator="or"):
    """
    Επεξεργάζεται απλά queries (χωρίς Boolean operators).
    """
    result_set = set()
    for i, token in enumerate(tokens):
        docs_for_token = set(map(int, inverted_index.get(token, [])))

        if i == 0:
            result_set = docs_for_token
        else:
            if default_operator == "or":
                result_set |= docs_for_token
            elif default_operator == "and":
                result_set &= docs_for_token

    return result_set

def process_query(query, inverted_index, default_operator="or"):
    """
    Ενιαία συνάρτηση: αν έχει AND/OR/NOT, κάνει Boolean search,
    αλλιώς κάνει simple search (OR/AND) προεπιλεγμένη.
    """
    tokens = query.lower().split()
    has_boolean_ops = any(tok in ["and", "or", "not"] for tok in tokens)

    if has_boolean_ops:
        # Προεπεξεργασία: μετατρέπουμε το query σε stemmed tokens
        stemmed_tokens = preprocess_query(query)
        return process_boolean_query(stemmed_tokens, inverted_index)
    else:
        # Προεπεργασία για simple query
        stemmed_tokens = preprocess_query(query)
        return process_simple_query(stemmed_tokens, inverted_index, default_operator)

def main():
    # Αρχεία
    index_file = "inverted_index.json"  # Απλό inverted index (χωρίς TF-IDF)
    docs_file = "wiki_definitions_cleaned.json"  # Αρχείο με τα έγγραφα

    # Φόρτωση Inverted Index
    inverted_index = load_inverted_index(index_file)
    if not inverted_index:
        return

    # Φόρτωση Εγγράφων (με paragraphs)
    documents = load_documents(docs_file)
    if not documents:
        return

    print("Welcome to the Search Engine!")
    print("You can use Boolean operators (AND, OR, NOT) or just words.")
    print("Examples:")
    print("  machine AND learning")
    print("  machine OR learning NOT deep")
    print("  machine learning deep (default OR)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        # Επεξεργασία ερωτήματος
        result_docs = process_query(query, inverted_index, default_operator="or")

        if result_docs:
            print(f"\nFound {len(result_docs)} documents:")
            for doc_id in sorted(result_docs):
                real_index = doc_id - 1  # αν doc_id ξεκινά από 1
                if 0 <= real_index < len(documents):
                    doc = documents[real_index]
                    title = doc.get('title', 'No Title')
                    original_def = doc.get('original_definition', 'No Definition')

                    print(f"  Document ID: {doc_id}")
                    print(f"    Title: {title}")
                    snippet = original_def[:200] + "..." if len(original_def) > 200 else original_def
                    print(f"    Definition: {snippet}\n")
                else:
                    # Αν για κάποιο λόγο το doc_id δεν αντιστοιχεί στην λίστα:
                    print(f"  Document ID: {doc_id} (not found in documents list)\n")
        else:
            print("No results found.")

if __name__ == "__main__":
        main()
